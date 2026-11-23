"""Mixed-mode log-magnitude curves and PLTS-compatible CSV export."""
import csv
import numpy as np
from .core import MIX

PARAMETERS = ('SDD11', 'SDD12', 'SDD21', 'SDD22', 'SCC11', 'SCC12', 'SCC21', 'SCC22')


def curves(network, ports=(1, 2, 3, 4)):
    """Ports are input+, output+, input-, output- (one-based)."""
    if network.nports != 4 or sorted(ports) != [1, 2, 3, 4]:
        raise ValueError('需要四端口 S4P 文件及四个不重复的端口号')
    z = np.asarray(network.z0)
    if not np.allclose(z, z[:, :1]) or not np.allclose(z.imag, 0) or np.any(z.real <= 0):
        raise ValueError('混模转换要求所有端口具有相同的正实数参考阻抗')
    order = np.asarray(ports) - 1
    mixed = MIX @ network.s[:, order][:, :, order] @ MIX.T
    result = {}
    for mode, offset in [('SDD', 0), ('SCC', 2)]:
        for i in range(2):
            for j in range(2):
                with np.errstate(divide='ignore', invalid='ignore'):
                    result[f'{mode}{i+1}{j+1}'] = 20*np.log10(np.abs(mixed[:, offset+i, offset+j]))
    return result


def write_curve(path, frequency_hz, name, values):
    if name not in PARAMETERS: raise ValueError('未知混模参数')
    if len(frequency_hz) != len(values): raise ValueError('频率与曲线数据长度不一致')
    with open(path, 'w', newline='', encoding='utf-8-sig') as fh:
        writer = csv.writer(fh, lineterminator='\n')
        writer.writerow(['Frequency(MHz)', f'{name}(Log Mag)'])
        for frequency, value in zip(frequency_hz, values):
            writer.writerow([f'{frequency/1e6:.15f}', f'{value:.15f}'])
