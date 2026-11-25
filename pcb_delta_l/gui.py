import csv
import sys
import traceback
from pathlib import Path
import numpy as np
import skrf as rf
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (QApplication,QMainWindow,QWidget,QVBoxLayout,QHBoxLayout,QFormLayout,QLabel,QPushButton,QLineEdit,QDoubleSpinBox,QSpinBox,QFileDialog,QMessageBox,QTabWidget,QSplitter,QScrollArea,QPlainTextEdit,QComboBox)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib import rcParams, font_manager

# Matplotlib does not inherit the Qt application font. Pick an installed CJK
# font explicitly so Chinese plot titles work on Windows, macOS, and Linux.
_CJK_FONTS = ("Microsoft YaHei", "SimHei", "PingFang SC", "Heiti SC",
              "Noto Sans CJK SC", "WenQuanYi Micro Hei", "Arial Unicode MS")
_installed_fonts = {font.name for font in font_manager.fontManager.ttflist}
rcParams["font.sans-serif"] = [name for name in _CJK_FONTS if name in _installed_fonts] + rcParams["font.sans-serif"]
rcParams["axes.unicode_minus"] = False
from .core import extract_2l
from .fitting import fit_loss
from .simulation import StressOptions, StressSamples, generate, generate_stress
from .touchstone import write_s4p, length_filename, MAX_PLTS_POINTS
from .comparison import DeltaLData, read_delta_l, write_plts, align
from .mixed_mode import curves as mixed_curves, write_curve, PARAMETERS

_INCHES_PER_UNIT = {'in': 1., 'mm': 1 / 25.4, 'ft': 12.}
_GLITCH_FIELDS = ('glitch_ghz', 'glitch_db', 'glitch_phase_deg')
_STRESS_PRESETS = {
    '005': ('P3+T1 探针不对称', ('launch_skew_ps',)),
    '006': ('P4+T1 单点毛刺', _GLITCH_FIELDS),
    '007': ('P1+T2 高损耗加噪声', ('noise_start_ghz', 'noise_floor_db', 'noise_seed')),
    '009': ('P1+T3 周期阻抗波动', ('ripple_ohm', 'period_in')),
    '011': ('P4+T4 偏斜加毛刺', ('skew_ps_per_in', *_GLITCH_FIELDS)),
}
_STRESS_FIELDS = {
    'launch_skew_ps': ('每端 P/N 时延差 (ps)', .001, 100., 3),
    'glitch_ghz': ('毛刺目标频率 (GHz)', .000000001, 1000., 9),
    'glitch_db': ('毛刺附加衰减 (dB)', 0., 60., 2),
    'glitch_phase_deg': ('毛刺附加相位 (°)', -180., 180., 2),
    'noise_start_ghz': ('噪声起始频率 (GHz)', 0., 1000., 3),
    'noise_floor_db': ('复噪声 RMS 电平 (dB)', -150., -1., 1),
    'noise_seed': ('随机种子', 0, 2147483647, 0),
    'ripple_ohm': ('周期阻抗波动幅度 (Ω)', .001, 999., 3),
    'period_in': ('阻抗波动周期 (in)', .001, 100., 3),
    'skew_ps_per_in': ('单位长度 P/N 时延差 (ps/in)', .001, 100., 3),
}

class Worker(QThread):
    success=pyqtSignal(object); failure=pyqtSignal(str)
    def __init__(self, fn): super().__init__(); self.fn=fn
    def run(self):
        try: self.success.emit(self.fn())
        except Exception: self.failure.emit(traceback.format_exc())

class Plot(QWidget):
    def __init__(self):
        super().__init__(); layout=QVBoxLayout(self); self.fig=Figure(figsize=(7,5),tight_layout=True); self.canvas=FigureCanvasQTAgg(self.fig); layout.addWidget(NavigationToolbar2QT(self.canvas,self)); layout.addWidget(self.canvas)
    def draw(self, f, curves, ylabel):
        self.fig.clear(); ax=self.fig.subplots();
        for y,label,style in curves: ax.plot(f/1e9,y,style,label=label,linewidth=1.5)
        ax.set_xlabel('Frequency (GHz)'); ax.set_ylabel(ylabel); ax.grid(True,color='#e7ecf3'); ax.legend(); self.canvas.draw_idle()

    def draw_result(self, raw, fit):
        self.fig.clear(); ax=self.fig.subplots(); other=ax.twinx(); f=raw.frequency_hz/1e9
        a,=ax.plot(f,fit.displayed_loss_db_per_in,color='#2563eb',label='Eigen Value')
        b,=ax.plot(f,fit.fitted_db_per_in,color='#dc2626',label='Curve Fitted')
        c,=other.plot(f,fit.uncertainty_percent,color='#d97706',linestyle='--',label='Uncertainty (%)')
        ax.set_xlabel('Frequency (GHz)'); ax.set_ylabel('Loss (dB/in)'); other.set_ylabel('Uncertainty (%)')
        ax.grid(True,color='#e7ecf3'); ax.legend(handles=[a,b,c],loc='best'); self.canvas.draw_idle()

class Window(QMainWindow):
    def __init__(self):
        super().__init__(); self.setWindowTitle('PCB Delta-L Workbench'); self.resize(1400,850); self.worker=None; self.result=None; self.sim=None
        root=QWidget(); self.setCentralWidget(root); outer=QVBoxLayout(root); title=QLabel('PCB Delta-L 工作台'); title.setStyleSheet('font-size:20px;font-weight:600'); outer.addWidget(title)
        self.tabs=QTabWidget(); outer.addWidget(self.tabs,1); self.log=QPlainTextEdit(); self.log.setReadOnly(True); self.log.setMaximumHeight(85); outer.addWidget(self.log); self.statusBar().showMessage('就绪 · Delta-L 2L')
        self.sim_page(); self.calc_page(); self.compare_page(); self.mixed_page()
    def split(self):
        s=QSplitter(); panel=QWidget(); form=QVBoxLayout(panel); scroll=QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(panel); scroll.setMinimumWidth(315); plot=Plot(); s.addWidget(scroll); s.addWidget(plot); s.setSizes([360,1000]); return s,form,plot
    def number(self, value, lo, hi, decimals=3):
        x=QDoubleSpinBox(); x.setRange(lo,hi); x.setDecimals(decimals); x.setValue(value); return x
    def button(self, text, slot, box):
        b=QPushButton(text); b.clicked.connect(slot); box.addWidget(b); return b
    def sim_page(self):
        s,box,self.simplot=self.split(); self.tabs.addTab(s,'仿真测试件'); form=QFormLayout(); box.addLayout(form)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.start=self.number(.01,0.000000001,1e12,9); self.stop=self.number(40,0.000000001,1e12,9); self.points=QSpinBox(); self.points.setRange(2,MAX_PLTS_POINTS); self.points.setValue(1601)
        self.lengths=QLineEdit('2, 5, 10'); self.z=self.number(95,1,1000); self.er=self.number(3.5,.1,30); self.skin=self.number(.08,0,100,5); self.diel=self.number(.035,0,100,5); self.fz=self.number(85,1,1000); self.fl=self.number(.15,0,10)
        self.preset=QComboBox()
        self.preset.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.preset.setMinimumContentsLength(17)
        self.preset.addItem('原有默认')
        self.preset.addItem('P1 + T1 低反射基准', (100, None))
        self.preset.addItem('P2 + T1 失配引入', (60, None))
        self.preset.addItem('P1 + T3 长短线阻抗差', (100, 102))
        self.preset.addItem('P2 + T3 失配引入 + 长短线阻抗差', (60, 103))
        for case, (title, _) in _STRESS_PRESETS.items():
            self.preset.addItem(f'TC-2L-{case} {title}', case)
        self.length_unit=QComboBox(); self.length_unit.addItems(['in','mm','ft']); self._length_unit='in'
        self.freq_unit=QComboBox(); self.freq_unit.addItems(['GHz','MHz','Hz']); self._freq_unit='GHz'
        self.long_z=self.number(102,1,1000)
        self.long_z.setEnabled(False)
        form.addRow('测试样本',self.preset); form.addRow('长度单位',self.length_unit); form.addRow('频率单位',self.freq_unit)
        for label,w in [('起始频率',self.start),('终止频率',self.stop),('频点数',self.points),('线长（逗号分隔）',self.lengths),('差模线阻抗 (Ω)',self.z),('有效介电常数',self.er),('集肤损耗系数',self.skin),('介质损耗系数',self.diel),('引入结构阻抗 (Ω)',self.fz),('单端引入长度 (in)',self.fl),('最长线差模阻抗 (Ω，仅 T3)',self.long_z)]: form.addRow(label,w)
        self.sim_form = form
        self.stress_controls = {}
        defaults = StressOptions()
        for key, (label, low, high, decimals) in _STRESS_FIELDS.items():
            if key == 'noise_seed':
                control = QSpinBox()
                control.setRange(low, high)
                control.setValue(defaults.noise_seed)
            else:
                control = self.number(getattr(defaults, key), low, high, decimals)
            form.addRow(label, control)
            form.setRowVisible(control, False)
            control.valueChanged.connect(self.invalidate_sim)
            self.stress_controls[key] = control
        self.sim_note = QLabel('仿真使用均匀传输线和相同引入结构。')
        self.sim_note.setWordWrap(True)
        box.addWidget(self.sim_note)
        self.button('生成仿真数据',self.run_sim,box); self.sim_save=self.button('导出 S4P',self.save_sim,box); self.sim_save.setEnabled(False); box.addStretch()
        for w in (self.start,self.stop,self.points,self.z,self.er,self.skin,self.diel,self.fz,self.fl): w.valueChanged.connect(self.invalidate_sim)
        self.lengths.textChanged.connect(self.invalidate_sim)
        self.long_z.valueChanged.connect(self.invalidate_sim)
        self.preset.currentIndexChanged.connect(self.apply_sim_preset)
        self.length_unit.currentTextChanged.connect(self.change_length_unit)
        self.freq_unit.currentTextChanged.connect(self.change_freq_unit)
    def apply_sim_preset(self):
        preset = self.preset.currentData()
        case = preset if isinstance(preset, str) else None
        fields = _STRESS_PRESETS[case][1] if case else ()
        for key, control in self.stress_controls.items():
            self.sim_form.setRowVisible(control, key in fields)
        self.sim_form.setRowVisible(self.long_z, case is None)
        self.long_z.setEnabled(isinstance(preset, tuple) and preset[1] is not None)
        if case:
            factor = _INCHES_PER_UNIT[self.length_unit.currentText()]
            self.lengths.setText(', '.join(format(length / factor, '.12g') for length in (5., 10.)))
            self.z.setValue(100)
            self.fz.setValue(100)
            self.fl.setValue(.15)
        elif preset is not None:
            fixture_z, longest_z = preset
            self.z.setValue(100)
            self.fz.setValue(fixture_z)
            self.fl.setValue(.15)
            if longest_z is not None:
                self.long_z.setValue(longest_z)
        notes = {
            '005': '长短线共用相同的 P/N 不对称引入。',
            '006': '仅在最长线最接近目标频率的点注入毛刺。_ref 文件为无毛刺参考。',
            '007': '材料损耗在 40 GHz 追加 2 dB/in。_ref 文件保留高损耗，且无噪声。',
            '009': '长短线使用相同幅度、周期和起点相位的阻抗波动。',
            '011': '偏斜随线长增加，毛刺只加在最长线。_ref 文件保留偏斜，且无毛刺。',
        }
        self.sim_note.setText(notes.get(case, '仿真使用均匀传输线和相同引入结构。'))
        self.invalidate_sim()
    def change_length_unit(self, unit):
        factors = _INCHES_PER_UNIT
        try:
            values=[float(x.strip()) for x in self.lengths.text().split(',')]
            values=[v*factors[self._length_unit]/factors[unit] for v in values]
            self.lengths.setText(', '.join(format(v,'.12g') for v in values))
            self._length_unit=unit
        except ValueError:
            self.length_unit.blockSignals(True); self.length_unit.setCurrentText(self._length_unit); self.length_unit.blockSignals(False)
            QMessageBox.warning(self,'单位切换失败','请先修正线长输入')
    def change_freq_unit(self, unit):
        factors={'GHz':1e9,'MHz':1e6,'Hz':1.}
        ratio=factors[self._freq_unit]/factors[unit]
        for control in (self.start,self.stop):
            control.blockSignals(True); control.setValue(control.value()*ratio); control.blockSignals(False)
        self._freq_unit=unit; self.invalidate_sim()
    def invalidate_sim(self,*_):
        self.sim=None; self.sim_save.setEnabled(False); self.simplot.fig.clear(); self.simplot.canvas.draw_idle()
    def calc_page(self):
        s,box,self.plot=self.split(); self.tabs.addTab(s,'Delta-L 计算'); f=QFormLayout(); box.addLayout(f); self.short=QLineEdit(); self.long=QLineEdit()
        for label,field in [('短线文件',self.short),('长线文件',self.long)]:
            row=QWidget(); h=QHBoxLayout(row); h.setContentsMargins(0,0,0,0); h.addWidget(field); b=QPushButton('选择'); b.clicked.connect(lambda _,w=field:self.browse(w)); h.addWidget(b); f.addRow(label,row)
        self.ls=self.number(2,0,10000); self.ll=self.number(5,0,10000); self.ports=QLineEdit('1,2,3,4')
        f.addRow('短线长度 (in)',self.ls); f.addRow('长线长度 (in)',self.ll); f.addRow('端口：输入+、输出+、输入−、输出−',self.ports)
        self.cutoff=self.number(0,0,1000); self.cutoff.setSpecialValueText('全频段'); self.weight=QComboBox(); self.weight.addItem('0 · 均匀权重',0); self.weight.addItem('3 · 低频加权',3)
        f.addRow('拟合截止频率 (GHz)',self.cutoff); f.addRow('拟合权重阶数',self.weight)
        self.fit_exponent = QDoubleSpinBox()
        self.fit_exponent.setDecimals(6)
        self.fit_exponent.setRange(0, 0.999999)
        self.fit_exponent.setSingleStep(0.01)
        self.fit_exponent.setSpecialValueText('自动')
        self.fit_exponent.setToolTip('输入对照样本的已知指数。TC-2L-007 使用 0.540625。')
        f.addRow('拟合指数 b', self.fit_exponent)
        note=QLabel('频率网格须一致。Uncertainty 使用 ±1 GHz 残差窗口。频带边缘缺少完整窗口时显示 0。'); note.setWordWrap(True); box.addWidget(note)
        self.button('计算 2L',self.run_calc,box); self.export=self.button('导出计算结果 CSV',self.save_result,box); self.export.setEnabled(False); box.addStretch()
        for w in (self.short,self.long,self.ports): w.textChanged.connect(self.invalidate)
        for w in (self.ls,self.ll,self.cutoff,self.fit_exponent): w.valueChanged.connect(self.invalidate)
        self.weight.currentIndexChanged.connect(self.invalidate)
    def invalidate(self,*_): self.result=None; self.export.setEnabled(False); self.plot.fig.clear(); self.plot.canvas.draw_idle()
    def browse(self,field):
        p,_=QFileDialog.getOpenFileName(self,'选择 Touchstone 文件','','Touchstone (*.s2p *.s4p)');
        if p: field.setText(p)
    def launch(self,fn,done):
        if self.worker and self.worker.isRunning(): return
        self.worker=Worker(fn); self.tabs.setEnabled(False); self.statusBar().showMessage('计算中…'); self.worker.success.connect(done); self.worker.failure.connect(self.fail); self.worker.finished.connect(lambda:self.tabs.setEnabled(True)); self.worker.start()
    def fail(self,detail): self.log.appendPlainText(detail); self.statusBar().showMessage('操作失败'); QMessageBox.warning(self,'操作失败',detail.strip().splitlines()[-1])
    def run_sim(self):
        try:
            lengths=tuple(float(x.strip())*_INCHES_PER_UNIT[self.length_unit.currentText()] for x in self.lengths.text().split(',')); factor={'GHz':1e9,'MHz':1e6,'Hz':1.}[self.freq_unit.currentText()]; start=self.start.value()*factor; stop=self.stop.value()*factor
            if len(lengths)<2 or len(set(lengths))!=len(lengths) or stop<=start: raise ValueError('至少需要两个不同长度，且终止频率必须大于起始频率')
            f = np.linspace(start, stop, self.points.value())
            preset = self.preset.currentData()
            case = preset if isinstance(preset, str) else None
            zmap = {}
            if isinstance(preset, tuple) and preset[1] is not None:
                zmap[max(lengths)] = self.long_z.value()
            line_options = dict(z_line=self.z.value(), er=self.er.value(),
                                skin=self.skin.value(), dielectric=self.diel.value(),
                                fixture_z=self.fz.value(), fixture_length=self.fl.value(),
                                length_impedances=zmap)
            if case:
                fields = _STRESS_PRESETS[case][1]
                options = StressOptions(**{key: self.stress_controls[key].value() for key in fields})
        except ValueError as e: QMessageBox.warning(self,'输入错误',str(e)); return
        self.sim=None; self.sim_save.setEnabled(False)
        def work():
            if case:
                return generate_stress(f, lengths, case, options, **line_options)
            networks, truth = generate(f, lengths, **line_options)
            return StressSamples(networks, {}, truth, [])
        def done(data):
            self.sim = data
            label = 'Material loss' if case else 'True loss'
            self.simplot.draw(f, [(data.material_loss, label, 'b-')], 'Loss (dB/in)')
            self.sim_save.setEnabled(True)
            self.statusBar().showMessage(f'已生成 {len(data.networks)} 个测试件、{len(data.reference)} 个参考件'
                                         if data.reference else f'已生成 {len(data.networks)} 个测试件')
            self.log.appendPlainText('仿真生成完成；S4P 端口顺序：输入+、输出+、输入−、输出−')
            for note in data.notes:
                self.log.appendPlainText(f'TC-2L-{case} · {note}')
        self.launch(work,done)
    def save_sim(self):
        folder=QFileDialog.getExistingDirectory(self,'选择输出目录')
        if not folder or self.sim is None:return
        try:
            preset = self.preset.currentData()
            prefix = f'tc_2l_{preset}_' if isinstance(preset, str) else ''
            for networks, suffix in ((self.sim.networks, ''), (self.sim.reference, '_ref')):
                for length, net in networks.items():
                    filename = f'{prefix}{Path(length_filename(length)).stem}{suffix}.s4p'
                    write_s4p(Path(folder) / filename, net)
            count = len(self.sim.networks) + len(self.sim.reference)
            self.statusBar().showMessage(f'已导出 {count} 个 S4P 文件')
        except Exception as e: QMessageBox.warning(self,'导出失败',str(e))
    def run_calc(self):
        try:
            paths=(self.short.text().strip(),self.long.text().strip()); ports=tuple(int(x.strip()) for x in self.ports.text().split(',')); ls=self.ls.value(); ll=self.ll.value()
            if not all(paths): raise ValueError('请选择长线和短线文件')
            cutoff=self.cutoff.value()*1e9 or None; weight=self.weight.currentData()
            exponent = self.fit_exponent.value() or None
        except ValueError as e: QMessageBox.warning(self,'输入错误',str(e)); return
        self.invalidate()
        def work():
            raw=extract_2l(rf.Network(paths[0]),rf.Network(paths[1]),ls,ll,ports)
            fit=fit_loss(raw.frequency_hz,raw.loss_db_per_in,raw.valid,cutoff,weight,exponent=exponent)
            return raw,fit
        def done(data):
            raw,fit=data; self.result=data; self.export.setEnabled(True); self.plot.draw_result(raw,fit)
            self.statusBar().showMessage(f'有效频点 {sum(raw.valid)}/{len(raw.valid)}；拟合 RMS {fit.rms_db_per_in:.4g} dB/in')
            self.log.appendPlainText(f'2L 完成；拟合系数 (a,b,c,d,offset)={fit.coefficients}，权重阶数={fit.weight_order}')
        self.launch(work,done)
    def save_result(self):
        if self.result is None:return
        p,_=QFileDialog.getSaveFileName(self,'导出结果','delta_l.csv','CSV (*.csv)')
        if not p:return
        r,fit=self.result
        try:
            write_plts(p, r, fit)
            self.statusBar().showMessage('结果已按 PLTS 格式导出，横轴数值为 MHz')
        except OSError as e: QMessageBox.warning(self,'导出失败',str(e))

    def compare_page(self):
        panel,box,self.compare_plot=self.split(); self.tabs.addTab(panel,'Delta-L 结果比较')
        self.compare_paths=[]; self.compare_units=[]
        for name in ('参考 A','对比 B'):
            box.addWidget(QLabel(name))
            row=QHBoxLayout(); field=QLineEdit(); field.setPlaceholderText('选择 PLTS 或程序结果 CSV'); row.addWidget(field)
            button=QPushButton('选择'); button.clicked.connect(lambda checked, w=field:self.browse_csv(w)); row.addWidget(button); box.addLayout(row)
            unit=QComboBox(); unit.addItems(['MHz','GHz','kHz','Hz']); unit.setToolTip('PLTS 风格 CSV 第一列的实际频率单位；旧版程序 CSV 自动使用 Hz')
            form=QFormLayout(); form.addRow('横轴频率单位',unit); box.addLayout(form)
            self.compare_paths.append(field); self.compare_units.append(unit)
        note=QLabel('按参考 A 的频点比较，B 在线性插值后相减（B − A），不外推。PLTS 表头 Time(ns) 的真实单位请根据数据确认。旧版程序导出的百分比是单点残差。')
        note.setWordWrap(True); box.addWidget(note)
        self.button('比较两个 CSV',self.run_compare,box)
        self.button('比较参考 A 与当前计算结果',self.compare_current,box)
        self.compare_summary=QPlainTextEdit(); self.compare_summary.setReadOnly(True); self.compare_summary.setMinimumHeight(155); box.addWidget(self.compare_summary); box.addStretch()
    def browse_csv(self,field):
        p,_=QFileDialog.getOpenFileName(self,'选择 Delta-L 结果 CSV','','CSV (*.csv);;所有文件 (*)')
        if p: field.setText(p)
    def run_compare(self):
        try:
            a=read_delta_l(self.compare_paths[0].text().strip(),self.compare_units[0].currentText())
            b=read_delta_l(self.compare_paths[1].text().strip(),self.compare_units[1].currentText())
            self.show_comparison(a,b)
        except (OSError, ValueError) as e: QMessageBox.warning(self,'比较失败',str(e))
    def compare_current(self):
        try:
            if self.result is None: raise ValueError('请先在 Delta-L 计算页面完成计算')
            a=read_delta_l(self.compare_paths[0].text().strip(),self.compare_units[0].currentText())
            raw,fit=self.result
            b=DeltaLData(raw.frequency_hz,np.column_stack((fit.displayed_loss_db_per_in,fit.fitted_db_per_in,fit.uncertainty_percent)))
            self.show_comparison(a,b)
        except (OSError, ValueError) as e: QMessageBox.warning(self,'比较失败',str(e))
    def show_comparison(self,a,b):
        aligned=align(a,b); delta=aligned-a.values
        if not np.any(np.isfinite(delta)): raise ValueError('两个结果没有可比较的有效频点，请检查频率单位和数据')
        fig=self.compare_plot.fig; fig.clear(); axes=fig.subplots(3,2); labels=['Eigen Value (dB/in)','Curve Fitted (dB/in)','Percent (%)']
        lines=[]
        for i,label in enumerate(labels):
            f=a.frequency_hz/1e9; valid=np.isfinite(delta[:,i]); ax=axes[i,0]; diff=axes[i,1]
            ax.plot(f,a.values[:,i],label='A',color='#2563eb'); ax.plot(f,aligned[:,i],label='B',color='#e11d48',alpha=.8); ax.set_ylabel(label); ax.legend(loc='best')
            diff.plot(f,delta[:,i],color='#7c3aed'); diff.axhline(0,color='#94a3b8',linewidth=.8); diff.set_ylabel('B − A')
            for x in (ax,diff): x.grid(True,color='#e7ecf3'); x.set_xlabel('Frequency (GHz)')
            if np.any(valid):
                d=delta[valid,i]; idx=np.flatnonzero(valid)[np.argmax(np.abs(d))]
                lines.append(f'{label}: N={len(d)}, 最大绝对差={np.max(np.abs(d)):.6g}，RMS={np.sqrt(np.mean(d*d)):.6g}，最大差频率={f[idx]:.6g} GHz')
            else: lines.append(f'{label}: 无有效重叠点')
        axes[0,0].set_title('曲线叠加'); axes[0,1].set_title('差值 B − A')
        self.compare_plot.canvas.draw_idle(); self.compare_summary.setPlainText('\n'.join(lines)); self.statusBar().showMessage('比较完成；百分比指标定义可能不同')
    def mixed_page(self):
        panel,box,self.mixed_plot=self.split(); self.tabs.addTab(panel,'混模 S 参数')
        self.mixed_files=[]; self.mixed_data=None
        form=QFormLayout(); box.addLayout(form)
        for label in ('文件 A','文件 B'):
            row=QWidget(); h=QHBoxLayout(row); h.setContentsMargins(0,0,0,0); field=QLineEdit(); h.addWidget(field)
            button=QPushButton('选择'); button.clicked.connect(lambda checked,w=field:self.browse(w)); h.addWidget(button)
            form.addRow(label,row); self.mixed_files.append(field); field.textChanged.connect(self.clear_mixed)
        self.mixed_ports=QLineEdit('1,2,3,4'); self.mixed_ports.textChanged.connect(self.clear_mixed)
        form.addRow('端口：输入+、输出+、输入−、输出−',self.mixed_ports)
        note=QLabel('每个四端口文件显示 SDD 和 SCC 各 4 条曲线。文件 B 可选；两文件共 16 条。')
        note.setWordWrap(True); box.addWidget(note)
        self.button('加载并显示',self.load_mixed,box)
        self.mixed_object=QComboBox(); self.mixed_object.addItems(['文件 A','文件 B']); self.mixed_parameter=QComboBox(); self.mixed_parameter.addItems(PARAMETERS)
        form2=QFormLayout(); form2.addRow('导出对象',self.mixed_object); form2.addRow('导出曲线',self.mixed_parameter); box.addLayout(form2)
        self.mixed_save=self.button('保存当前曲线 CSV',self.save_mixed_curve,box)
        self.mixed_save_all=self.button('批量保存全部曲线 CSV',self.save_all_mixed,box)
        self.mixed_save.setEnabled(False); self.mixed_save_all.setEnabled(False); box.addStretch()
    def clear_mixed(self,*args):
        self.mixed_data=None
        if hasattr(self,'mixed_save'): self.mixed_save.setEnabled(False); self.mixed_save_all.setEnabled(False)
        if hasattr(self,'mixed_plot'): self.mixed_plot.fig.clear(); self.mixed_plot.canvas.draw_idle()
    def load_mixed(self):
        try:
            paths=[w.text().strip() for w in self.mixed_files]; ports=tuple(int(x.strip()) for x in self.mixed_ports.text().split(','))
            if not paths[0]: raise ValueError('请选择文件 A')
            datasets=[]
            for path in paths:
                if not path: datasets.append(None); continue
                net=rf.Network(path); datasets.append((net.f.copy(),mixed_curves(net,ports),path))
            self.mixed_data=datasets; self.mixed_save.setEnabled(True); self.mixed_save_all.setEnabled(True)
            fig=self.mixed_plot.fig; fig.clear(); axes=fig.subplots(4,4 if datasets[1] is not None else 2, squeeze=False)
            order=('SDD11','SDD21','SDD12','SDD22','SCC11','SCC21','SCC12','SCC22')
            for k,data in enumerate(datasets):
                if data is None: continue
                f,items,path=data
                for idx,name in enumerate(order):
                    ax=axes[idx//2,k*2+idx%2]; ax.plot(f/1e9,items[name],color='#2563eb',linewidth=1)
                    ax.set_title(f'{chr(65+k)} · {name}',fontsize=9); ax.grid(True,color='#e7ecf3'); ax.set_ylabel('dB',fontsize=8)
                    if idx//2==3: ax.set_xlabel('GHz',fontsize=8)
                    ax.tick_params(labelsize=7)
            self.mixed_plot.canvas.draw_idle(); self.statusBar().showMessage('混模曲线已加载')
        except (OSError,ValueError) as e: self.clear_mixed(); QMessageBox.warning(self,'加载失败',str(e))
    def save_mixed_curve(self):
        if self.mixed_data is None: return
        idx=self.mixed_object.currentIndex(); data=self.mixed_data[idx]
        if data is None: QMessageBox.warning(self,'导出失败','请先选择并加载文件 B'); return
        name=self.mixed_parameter.currentText(); p,_=QFileDialog.getSaveFileName(self,'保存当前曲线',name+'.csv','CSV (*.csv)')
        if not p: return
        try: write_curve(p,data[0],name,data[1][name]); self.statusBar().showMessage(f'{name} 已保存')
        except OSError as e: QMessageBox.warning(self,'导出失败',str(e))
    def save_all_mixed(self):
        if self.mixed_data is None: return
        folder=QFileDialog.getExistingDirectory(self,'选择曲线导出目录')
        if not folder: return
        paths=[Path(folder)/(f'{chr(65+i)}_{name}.csv') for i,data in enumerate(self.mixed_data) if data is not None for name in PARAMETERS]
        if any(p.exists() for p in paths) and QMessageBox.question(self,'文件已存在','目标目录包含同名曲线文件，是否覆盖？') != QMessageBox.StandardButton.Yes: return
        try:
            for i,data in enumerate(self.mixed_data):
                if data is not None:
                    for name in PARAMETERS: write_curve(Path(folder)/f'{chr(65+i)}_{name}.csv',data[0],name,data[1][name])
            self.statusBar().showMessage(f'已保存 {len(paths)} 条混模曲线')
        except OSError as e: QMessageBox.warning(self,'导出失败',str(e))
    def closeEvent(self,e):
        if self.worker and self.worker.isRunning(): QMessageBox.information(self,'请稍候','计算完成后可以关闭窗口。'); e.ignore()
        else:e.accept()

def main():
    app=QApplication(sys.argv); app.setStyleSheet('QWidget{font-family: Microsoft YaHei, Noto Sans CJK SC, sans-serif; color:#24344a; font-size:10pt} QMainWindow{background:#f3f6fa} QLineEdit,QSpinBox,QDoubleSpinBox,QPlainTextEdit{background:white;border:1px solid #cbd5e1;padding:5px;border-radius:4px} QPushButton{background:white;border:1px solid #cbd5e1;padding:7px;border-radius:4px} QPushButton:hover{border-color:#2563eb}'); w=Window(); w.show(); sys.exit(app.exec())
if __name__=='__main__':main()
