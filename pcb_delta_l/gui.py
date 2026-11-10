import csv
import sys
import traceback
from pathlib import Path
import numpy as np
import skrf as rf
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (QApplication,QMainWindow,QWidget,QVBoxLayout,QHBoxLayout,QFormLayout,QLabel,QPushButton,QLineEdit,QDoubleSpinBox,QSpinBox,QFileDialog,QMessageBox,QTabWidget,QSplitter,QScrollArea,QPlainTextEdit)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from .core import extract_2l
from .simulation import generate

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

class Window(QMainWindow):
    def __init__(self):
        super().__init__(); self.setWindowTitle('PCB Delta-L Workbench'); self.resize(1400,850); self.worker=None; self.result=None; self.sim=None
        root=QWidget(); self.setCentralWidget(root); outer=QVBoxLayout(root); title=QLabel('PCB Delta-L 工作台'); title.setStyleSheet('font-size:20px;font-weight:600'); outer.addWidget(title)
        self.tabs=QTabWidget(); outer.addWidget(self.tabs,1); self.log=QPlainTextEdit(); self.log.setReadOnly(True); self.log.setMaximumHeight(85); outer.addWidget(self.log); self.statusBar().showMessage('就绪 · 2L 特征值研究版，尚未与 PLTS 验证')
        self.sim_page(); self.calc_page()
    def split(self):
        s=QSplitter(); panel=QWidget(); form=QVBoxLayout(panel); scroll=QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(panel); scroll.setMinimumWidth(315); plot=Plot(); s.addWidget(scroll); s.addWidget(plot); s.setSizes([360,1000]); return s,form,plot
    def number(self, value, lo, hi, decimals=3):
        x=QDoubleSpinBox(); x.setRange(lo,hi); x.setDecimals(decimals); x.setValue(value); return x
    def button(self, text, slot, box):
        b=QPushButton(text); b.clicked.connect(slot); box.addWidget(b); return b
    def sim_page(self):
        s,box,self.simplot=self.split(); self.tabs.addTab(s,'仿真测试件'); form=QFormLayout(); box.addLayout(form)
        self.start=self.number(.01,.001,1000); self.stop=self.number(40,.002,1000); self.points=QSpinBox(); self.points.setRange(2,100001); self.points.setValue(1601)
        self.lengths=QLineEdit('2, 5, 10'); self.z=self.number(95,1,1000); self.er=self.number(3.5,.1,30); self.skin=self.number(.08,0,100,5); self.diel=self.number(.035,0,100,5); self.fz=self.number(85,1,1000); self.fl=self.number(.15,0,10)
        for label,w in [('起始频率 (GHz)',self.start),('终止频率 (GHz)',self.stop),('频点数',self.points),('线长 (in，逗号分隔)',self.lengths),('差模线阻抗 (Ω)',self.z),('有效介电常数',self.er),('集肤损耗系数',self.skin),('介质损耗系数',self.diel),('引入结构阻抗 (Ω)',self.fz),('单端引入长度 (in)',self.fl)]: form.addRow(label,w)
        note=QLabel('仿真使用均匀传输线和相同引入结构，默认无模式转换。'); note.setWordWrap(True); box.addWidget(note)
        self.button('生成仿真数据',self.run_sim,box); self.sim_save=self.button('导出 S4P 和真值',self.save_sim,box); self.sim_save.setEnabled(False); box.addStretch()
        for w in (self.start,self.stop,self.points,self.z,self.er,self.skin,self.diel,self.fz,self.fl): w.valueChanged.connect(self.invalidate_sim)
        self.lengths.textChanged.connect(self.invalidate_sim)
    def invalidate_sim(self,*_):
        self.sim=None; self.sim_save.setEnabled(False); self.simplot.fig.clear(); self.simplot.canvas.draw_idle()
    def calc_page(self):
        s,box,self.plot=self.split(); self.tabs.addTab(s,'Delta-L 计算'); f=QFormLayout(); box.addLayout(f); self.short=QLineEdit(); self.long=QLineEdit()
        for label,field in [('短线文件',self.short),('长线文件',self.long)]:
            row=QWidget(); h=QHBoxLayout(row); h.setContentsMargins(0,0,0,0); h.addWidget(field); b=QPushButton('选择'); b.clicked.connect(lambda _,w=field:self.browse(w)); h.addWidget(b); f.addRow(label,row)
        self.ls=self.number(2,0,10000); self.ll=self.number(5,0,10000); self.ports=QLineEdit('1,2,3,4')
        f.addRow('短线长度 (in)',self.ls); f.addRow('长线长度 (in)',self.ll); f.addRow('端口：输入+、输出+、输入−、输出−',self.ports)
        note=QLabel('频率网格须一致。输出原始特征值损耗；物理拟合和不确定度尚待 PLTS 数据验证。'); note.setWordWrap(True); box.addWidget(note)
        self.button('计算 2L',self.run_calc,box); self.export=self.button('导出计算结果 CSV',self.save_result,box); self.export.setEnabled(False); box.addStretch()
        for w in (self.short,self.long,self.ports): w.textChanged.connect(self.invalidate)
        for w in (self.ls,self.ll): w.valueChanged.connect(self.invalidate)
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
            lengths=tuple(float(x.strip()) for x in self.lengths.text().split(',')); start=self.start.value(); stop=self.stop.value()
            if len(lengths)<2 or len(set(lengths))!=len(lengths) or stop<=start: raise ValueError('至少需要两个不同长度，且终止频率必须大于起始频率')
            f=np.linspace(start*1e9,stop*1e9,self.points.value()); args=(self.z.value(),self.er.value(),self.skin.value(),self.diel.value(),self.fz.value(),self.fl.value())
        except ValueError as e: QMessageBox.warning(self,'输入错误',str(e)); return
        self.sim=None; self.sim_save.setEnabled(False)
        def work():
            z,er,skin,diel,fz,fl=args; return generate(f,lengths,z,er,0.01,skin,diel,fz,fl)
        def done(data):
            self.sim=data; nets,truth=data; self.simplot.draw(f,[(truth,'True loss','b-')],'Loss (dB/in)'); self.sim_save.setEnabled(True); self.statusBar().showMessage(f'已生成 {len(nets)} 个测试件'); self.log.appendPlainText('仿真生成完成；S4P 端口顺序：输入+、输出+、输入−、输出−')
        self.launch(work,done)
    def save_sim(self):
        folder=QFileDialog.getExistingDirectory(self,'选择输出目录')
        if not folder or self.sim is None:return
        nets,truth=self.sim
        try:
            for length,net in nets.items(): net.write_touchstone(str(Path(folder)/f'line_{length:g}in'))
            np.savetxt(Path(folder)/'truth.csv',np.column_stack((next(iter(nets.values())).f,truth)),delimiter=',',header='frequency_hz,loss_db_per_in',comments='')
            self.statusBar().showMessage('仿真 S4P 与真值已导出')
        except Exception as e: QMessageBox.warning(self,'导出失败',str(e))
    def run_calc(self):
        try:
            paths=(self.short.text().strip(),self.long.text().strip()); ports=tuple(int(x.strip()) for x in self.ports.text().split(',')); ls=self.ls.value(); ll=self.ll.value()
            if not all(paths): raise ValueError('请选择长线和短线文件')
        except ValueError as e: QMessageBox.warning(self,'输入错误',str(e)); return
        self.invalidate()
        def work(): return extract_2l(rf.Network(paths[0]),rf.Network(paths[1]),ls,ll,ports)
        def done(r):
            self.result=r; self.export.setEnabled(True); self.plot.draw(r.frequency_hz,[(r.loss_db_per_in,'Eigenvalue raw','b-')],'Loss (dB/in)'); self.statusBar().showMessage(f'计算完成：有效频点 {sum(r.valid)}/{len(r.valid)}'); self.log.appendPlainText(f'2L 完成：有效 {sum(r.valid)}/{len(r.valid)}；结果未经过 PLTS 验证')
        self.launch(work,done)
    def save_result(self):
        if self.result is None:return
        p,_=QFileDialog.getSaveFileName(self,'导出结果','delta_l.csv','CSV (*.csv)')
        if not p:return
        r=self.result
        try:
            with open(p,'w',newline='',encoding='utf-8-sig') as fh:
                w=csv.writer(fh); w.writerow(['frequency_hz','loss_db_per_in','eigenvalue_1_real','eigenvalue_1_imag','eigenvalue_2_real','eigenvalue_2_imag','short_abcd_condition','valid'])
                for f,l,e,c,v in zip(r.frequency_hz,r.loss_db_per_in,r.eigenvalues,r.condition,r.valid):w.writerow([f,l,e[0].real,e[0].imag,e[1].real,e[1].imag,c,int(v)])
            self.statusBar().showMessage('结果已导出')
        except OSError as e: QMessageBox.warning(self,'导出失败',str(e))
    def closeEvent(self,e):
        if self.worker and self.worker.isRunning(): QMessageBox.information(self,'请稍候','计算完成后可以关闭窗口。'); e.ignore()
        else:e.accept()

def main():
    app=QApplication(sys.argv); app.setStyleSheet('QWidget{font-family: Microsoft YaHei, Noto Sans CJK SC, sans-serif; color:#24344a; font-size:10pt} QMainWindow{background:#f3f6fa} QLineEdit,QSpinBox,QDoubleSpinBox,QPlainTextEdit{background:white;border:1px solid #cbd5e1;padding:5px;border-radius:4px} QPushButton{background:white;border:1px solid #cbd5e1;padding:7px;border-radius:4px} QPushButton:hover{border-color:#2563eb}'); w=Window(); w.show(); sys.exit(app.exec())
if __name__=='__main__':main()
