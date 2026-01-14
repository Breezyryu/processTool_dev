import os
import sys
import re
import bisect
import warnings
import json
import pyodbc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, root_scalar
from scipy.stats import linregress
from datetime import datetime
from tkinter import filedialog, Tk
from PyQt6 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from datetime import timezone
import glob
import xlwings as xw

# pip 추가 항목: xlsxwriter
# Malgun gothic을 기본 글꼴로 설정: %s/Malgun gothic/Malgun gothic/g

# 경고 무시
warnings.simplefilter("ignore")
# 한글 설정
plt.rcParams["font.family"] = "Malgun gothic"
plt.rcParams["axes.unicode_minus"] = False

# timestamp 변환 함수 정의
def to_timestamp(date_str):
    # 형식 파싱 (DDMMYY HH:MM:SS.msec)
    year = int(date_str[:2])
    month = int(date_str[2:4])
    day = int(date_str[4:6])
    hour = int(date_str[7:9])
    minute = int(date_str[10:12])
    second = int(date_str[13:15])
    millisecond = int(date_str[16:19])

    # 연도는 2000년대를 가정 (예: 17 -> 2017)
    year += 2000

    # datetime 객체로 변환
    dt = datetime(year, month, day, hour, minute, second, millisecond * 1000, tzinfo=timezone.utc)
    # Unix Timestamp로 변환 (초 단위)
    return int(dt.timestamp() - 9 * 3600)

# 진행 현황 만드는 함수 (점점 작은 범위로 산정)
def progress(count1, max1, count2, max2, count3, max3):
    progressdata = ((count1 + ((count2 + (count3 / max3) - 1) / max2) - 1) / max1 * 100)
    return progressdata

# 여러 directory 선택하는 코드
def multi_askopendirnames():
    directories = []
    while True:
        if len(directories) == 0:
            ini_dir = "d://"
        else:
            # 1. 경로 정규화
            normalized_path = os.path.normpath(directories[-1])
            # 2. 상위 디렉토리 추출
            parent_dir = os.path.dirname(normalized_path)
            # 3. 트레일링 백슬래시 추가
            ini_dir = os.path.join(parent_dir, "")  # 빈 문자열과 결합
        directory = filedialog.askdirectory(initialdir=ini_dir, title="원하는 폴더를 계속해서 선택하세요")
        if not directory:
            break
        directories.append(directory)
    return directories

# 대괄호 안의 문자 추출
def extract_text_in_brackets(input_string):
    # 정규 표현식으로 대괄호 안의 문자 추출
    match = re.search(r'\[(.*?)\]', input_string)
    return match.group(1) if match else str(input_string).zfill(3)

# 시리즈를 정해진 갯수의 column으로 분리하는 함수
def separate_series(df_series, num):
    """
    [시리즈 분리 함수]
    - df: 입력 DataFrame의 series
    - num: 분리할 열의 개수
    """
    # 결과 DataFrame 생성 (초기화)
    result_df = pd.DataFrame()
    # 각 값을 적절한 열에 매핑
    for i, value in enumerate(df_series, 1):
        # 열 이름 생성 (예: Column0, Column1, ...)
        col_name = f'col{(i - 1) % num}'
        # 행 인덱스 계산 (예: 0, 1, 2, ...)
        row_idx = (i - 1) // num
        # 값 할당
        result_df.at[row_idx, col_name] = value
    return result_df

# filepath 이름에서 용량을 추출하는 함수
def name_capacity(data_file_path):
    if not isinstance(data_file_path, list):
        # 원시 문자열을 사용하여 특수 문자를 공백으로 대체
        raw_file_path = re.sub(r'[._@\$$$$$$\(\)]', ' ', data_file_path)
        # 정규 표현식을 사용하여 "mAh"로 끝나는 용량 값을 찾습니다. (소수점 포함)
        match = re.search(r'(\d+([\-.]\d+)?)mAh', raw_file_path)
        # 소수점 용량을 위해 -를 .으로 변환
        if match:
            min_cap = match.group(1).replace('-', '.')
        # 일치하는 값이 있으면 실수로 변환하여 반환
            return float(min_cap)
        # 용량 값이 없으면 None을 반환하거나 오류를 발생시킵니다.
        return 0
    else:
        return 0

# 주어진 숫자가 리스트의 몇 번째에 해당하는지 확인하는 코드
def binary_search(numbers, target):
    index = bisect.bisect_left(numbers, target)
    return index

# csv의 마지막 ,를 제거하는 함수
def remove_end_comma(file_path):
    # 파일을 열고 내용을 확인
    with open(file_path, 'r', newline='', encoding='ANSI') as input_file:
        rows = input_file.read().splitlines()
    # 각 줄에 ,가 있으면 제거
    for i, row in enumerate(rows):
        if row.endswith(','):
            rows[i] = row[:-1]
    df = pd.DataFrame(row.split(',') for row in rows)
    return df

# error메세지 출력
def err_msg(title, msg):
    msg_box = QtWidgets.QMessageBox()
    msg_box.setWindowTitle(title)
    msg_box.setText(msg)
    msg_box.setIcon(QtWidgets.QMessageBox.Icon.Critical)
    # 폰트 설정
    font = QtGui.QFont()
    font.setFamily("Malgun gothic")
    msg_box.setFont(font)
    msg_box.exec()

# 연결되었을 경우 파란색으로 설정
def connect_change(button):
    color = QtGui.QColor(0, 0, 200)  # 파란색으로 설정
    button.setStyleSheet(f"color: {color.name()}")

# 연결 안 되었을 경우 빨간색으로 설정
def disconnect_change(button):
    color = QtGui.QColor(200, 0, 0)  # 빨간색으로 설정
    button.setStyleSheet(f"color: {color.name()}")

# 충방전기 구분 (패턴 폴더 유무로 구분)
def check_cycler(raw_file_path): 
    # 충방전기 데이터 폴더에 패턴 폴더 유무로 PNE와 Toyo 구분, PNE이면 True, Toyo이면 False
    cycler = os.path.isdir(raw_file_path + "\\Pattern")
    return cycler

# 주어진 문자열을 리스트로 변환
def convert_steplist(input_str):
    output_list = []
    for part in input_str.split():
        if "-" in part:
            start, end = map(int, part.split("-"))
            output_list.extend(range(start, end + 1))
        else:
            output_list.append(int(part))
    return output_list

# list에서 같은 같의 경우 1을 더해서 계산
def same_add(df, column_name):
    new_column_name = f"{column_name}_add"
    df[new_column_name] = df[column_name].apply(lambda x: x)
    # 중복된 값에 대해 1씩 증가
    df[new_column_name] = df.groupby(column_name)[new_column_name].cumcount().add(df[column_name])
    df[new_column_name] = df[new_column_name] - df[new_column_name].min() + 1
    return df
    
# 그래프 base 기본 설정 함수 (x라벨, y라벨, 그리드 양식)
def graph_base_parameter(graph_ax, xlabel, ylabel): 
    graph_ax.set_xlabel(xlabel, fontsize= 12, fontweight='bold')
    graph_ax.set_ylabel(ylabel, fontsize= 12, fontweight='bold')
    graph_ax.tick_params(direction='in')
    graph_ax.grid(True, which='both', linestyle='--', linewidth=1.0)

# Cycle 그래프 기본, x축, y축 min, max 및 범위 설정
def graph_cycle_base(x_data, ax, lowlimit, highlimit, y_gap, xlabel, ylabel, xscale, overall_xlimit):
    if xscale == 0 and len(x_data) != 0:
        xlimit = max(x_data)
        if xlimit < overall_xlimit:
            xlimit = overall_xlimit
        xrangemax = (xlimit // 100 + 2) * 100
    else:
        xlimit = xscale
        xrangemax = xscale
    xrangegap = ((xlimit >= 400) + (xlimit >= 800) * 2 + (xlimit >= 1200) * 4 + (xlimit >= 2000) * 2 + 1) * 50
    ax.set_xticks(np.arange(0, xrangemax + xrangegap, xrangegap))
    if highlimit != 0:
        ax.set_yticks(np.arange(lowlimit, highlimit, y_gap))
        ax.set_ylim(lowlimit, highlimit)
    graph_base_parameter(ax, xlabel, ylabel)

# Cycle 그래프 그리기 - 지정색 기준 사용
def graph_cycle(x, y, ax, lowlimt, highlimit, ygap, xlabel, ylabel, tlabel, xscale, cyc_color, overall_xlimit = 0):
    # 지정색이 없으면 기본색 사용
    if cyc_color != 0:
        ax.scatter(x, y, label=tlabel, s=5, color=cyc_color)
    else:
        ax.scatter(x, y, label=tlabel, s=5)
    graph_cycle_base(x, ax, lowlimt, highlimit, ygap, xlabel, ylabel, xscale, overall_xlimit = 0)    

# Cycle 그래프 그리기 - 지정색 기준 사용/ scatter 채우기 없음
def graph_cycle_empty(x, y, ax, lowlimt, highlimit, ygap, xlabel, ylabel, tlabel, xscale, cyc_color, overall_xlimit = 0):
    # 지정색이 없으면 기본색 사용
    if cyc_color != 0:
        ax.scatter(x, y, label=tlabel, s=8, edgecolors=cyc_color, facecolors ='none')
    else:
        ax.scatter(x, y, label=tlabel, s=8, facecolors = 'none')
    graph_cycle_base(x, ax, lowlimt, highlimit, ygap, xlabel, ylabel, xscale, overall_xlimit = 0)    

def graph_output_cycle(df, xscale, ylimitlow, ylimithigh, irscale, lgnd, temp_lgnd, colorno, graphcolor,
                       dcir, ax1, ax2, ax3, ax4, ax5, ax6):
    graph_cycle(df.NewData.index, df.NewData.Dchg, ax1, ylimitlow, ylimithigh, 0.05,
                "Cycle", "Discharge Capacity Ratio", temp_lgnd, xscale, graphcolor[colorno % 9])
    graph_cycle(df.NewData.index, df.NewData.Eff, ax2, 0.992, 1.004, 0.002,
                "Cycle", "Discharge/Charge Efficiency", temp_lgnd, xscale, graphcolor[colorno % 9])
    graph_cycle(df.NewData.index, df.NewData.Temp, ax3, 0, 50, 5,
                "Cycle", "Temperature (℃)", temp_lgnd, xscale, graphcolor[colorno % 9])
    graph_cycle(df.NewData.index, df.NewData.RndV, ax6, 3.00, 4.00, 0.1,
                "Cycle", "Rest End Voltage (V)", "", xscale, graphcolor[colorno % 9])
    graph_cycle_empty(df.NewData.index, df.NewData.Eff2, ax5, 0.996, 1.008, 0.002,
                      "Cycle", "Charge/Discharge Efficiency", temp_lgnd, xscale, graphcolor[colorno % 9])
    graph_cycle_empty(df.NewData.index, df.NewData.AvgV, ax6, 3.00, 4.00, 0.1,
                      "Cycle", "Average/Rest Voltage (V)", temp_lgnd, xscale, graphcolor[colorno % 9])
    if dcir.isChecked() and hasattr(df.NewData, "dcir2"):
        graph_cycle_empty(df.NewData.index, df.NewData.soc70_dcir, ax4, 0, 120.0 * irscale, 20 * irscale,
                        "Cycle", "RSS/ 1s DC-IR (mΩ)", "", xscale, graphcolor[colorno % 9])
        graph_cycle(df.NewData.index, df.NewData.soc70_rss_dcir, ax4, 0, 120.0 * irscale, 20 * irscale,
                    "Cycle", "RSS/ 1s DC-IR (mΩ)", temp_lgnd, xscale, graphcolor[colorno % 9])
    else:
        graph_cycle(df.NewData.index, df.NewData.dcir, ax4, 0, 120.0 * irscale, 20 * irscale,
                    "Cycle", "DC-IR (mΩ)", temp_lgnd, xscale, graphcolor[colorno % 9])
    colorno = colorno % 9 + 1

# Step charge Profile 그래프 그리기
def graph_step(x, y, ax, lowlimit, highlimit, limitgap, xlabel, ylabel, tlabel):
    ax.plot(x, y, label=tlabel)
    ax.set_yticks(np.arange(lowlimit, highlimit, limitgap))
    ax.set_ylim(lowlimit, highlimit - limitgap)
    graph_base_parameter(ax, xlabel, ylabel)

# 연속 그래프 그리기
def graph_continue(x, y, ax, lowlimit, highlimit, limitgap, xlabel, ylabel, tlabel, type = "-"):
    if type == "-":
        ax.plot(x, y, label=tlabel)
    else:
        ax.plot(x, y, label=tlabel, marker='o', markersize = 3)
    ax.set_yticks(np.arange(lowlimit, highlimit, limitgap))
    ax.set_ylim(lowlimit, highlimit - limitgap)
    graph_base_parameter(ax, xlabel, ylabel)

# 연속 그래프 그리기
def graph_soc_continue(x, y, ax, lowlimit, highlimit, limitgap, xlabel, ylabel, tlabel, type = "-"):
    if type == "-":
        ax.plot(x, y, label=tlabel)
    else:
        ax.plot(x, y, label=tlabel, marker='o', markersize = 3)
    ax.set_xticks(np.arange(0, 110, 10))
    ax.set_yticks(np.arange(lowlimit, highlimit, limitgap))
    ax.set_ylim(lowlimit, highlimit - limitgap)
    graph_base_parameter(ax, xlabel, ylabel)

# OCV 기반 DCIR 그래프 그리기
def graph_dcir(x, y, ax, xlabel, ylabel, tlabel, type = "-"):
    if type == "-":
        ax.plot(x, y, label=tlabel)
    else:
        ax.plot(x, y, label=tlabel, marker='o', markersize = 3)
    graph_base_parameter(ax, xlabel, ylabel)

# SOC별 DCIR 그래프 그리기
def graph_soc_dcir(x, y, ax, xlabel, ylabel, tlabel, type = "-"):
    if type == "-":
        ax.plot(x, y, label=tlabel)
    else:
        ax.plot(x, y, label=tlabel, marker='o', markersize = 3)
    ax.set_xticks(np.arange(0, 110, 10))
    graph_base_parameter(ax, xlabel, ylabel)

# 충방전 Profile 그래프 그리기
def graph_profile(x, y, ax, xlowlimit, xhighlimit, xlimitgap, ylowlimit, yhighlimit, ylimitgap, xlabel, ylabel, tlabel):
    ax.plot(x, y, label=tlabel)
    ax.set_xticks(np.arange(xlowlimit, xhighlimit, xlimitgap))
    ax.set_xlim(xlowlimit, xhighlimit - xlimitgap)
    ax.set_yticks(np.arange(ylowlimit, yhighlimit, ylimitgap))
    ax.set_ylim(ylowlimit, yhighlimit - ylimitgap)
    graph_base_parameter(ax, xlabel, ylabel)

# set profile 그래프 그리기
def graph_soc_set(x, y, ax, lowlimit, highlimit, limitgap, xlabel, ylabel, tlabel, xlimit):
    colors = {3: 'red', 4: 'blue', 5: 'green', 6: 'magenta', 7: 'cyan', 8: 'red', 9: 'red'}
    if xlimit == {0, 1, 2}:
        ax.scatter(x, y, label=tlabel, s=1)
    elif xlimit in colors:
        ax.scatter(x, y, label=tlabel, s=1, color = colors[xlimit])
    else:
        ax.scatter(x, y, label=tlabel, s=1)
    if limitgap != 0:
        ax.set_yticks(np.arange(lowlimit, highlimit, limitgap))
        ax.set_ylim(lowlimit, highlimit - limitgap)
    graph_base_parameter(ax, xlabel, ylabel)

# ECT SOC 에러 확인 그래프
def graph_soc_err(x, y, ax, lowlimit, highlimit, limitgap, xlabel, ylabel, tlabel, xlimit):
    colors = {3: 'red', 4: 'blue', 5: 'green', 6: 'magenta', 7: 'cyan', 8: 'red', 9: 'red'}
    df = pd.DataFrame({'x': x, 'y': abs(y)})
    grouped = df.groupby(df['x']//5).mean()
    index_x = grouped.index * 5
    ax.bar(index_x, grouped['y'], width = 4, align = 'center', label=tlabel, color = colors[xlimit], alpha=0.5)
    ax.set_yticks(np.arange(lowlimit, highlimit, limitgap))
    ax.set_ylim(lowlimit, highlimit - limitgap)
    ax.set_xlim(105, -5)
    ax.set_xticks(range(-5, 106, 5))
    graph_base_parameter(ax, xlabel, ylabel)

# set profile 그래프 그리기
def graph_set_profile(x, y, ax, y_llimit, y_hlimit, y_gap, xlabel, ylabel, tlabel, graphcolor, x_llimit, x_hlimit, x_gap):
    colors = {1: 'red', 2: 'blue', 3: 'green', 4: 'magenta', 5: 'cyan'}
    if graphcolor in colors:
        ax.scatter(x, y, label=tlabel, s=1, color=colors[graphcolor])
    else:
        ax.scatter(x, y, label=tlabel, s=1)
    if x_gap != 0:
        ax.set_xticks(np.arange(x_llimit, x_hlimit, x_gap))
        ax.set_xlim(x_llimit, x_hlimit - x_gap)
    if y_gap != 0:
        ax.set_yticks(np.arange(y_llimit, y_hlimit, y_gap))
        ax.set_ylim(y_llimit, y_hlimit - y_gap)
    graph_base_parameter(ax, xlabel, ylabel)

# SOC 산정을 위한 SOC 및 비교용 그래프
def graph_set_guide(x, y, ax, y_llimit, y_hlimit, y_gap, xlabel, ylabel, tlabel, x_llimit, x_hlimit, x_gap):
    ax.plot(x, y, linestyle = 'dotted', color = 'red', label=tlabel)
    if x_gap != 0:
        ax.set_xticks(np.arange(x_llimit, x_hlimit, x_gap))
        ax.set_xlim(x_llimit, x_hlimit - x_gap)
    if y_gap != 0:
        ax.set_yticks(np.arange(y_llimit, y_hlimit, y_gap))
        ax.set_ylim(y_llimit, y_hlimit - y_gap)
    graph_base_parameter(ax, xlabel, ylabel)

# Simulation graph
def graph_simulation(ax, x, y, pltcolor, pltlabel, x_limit, y_min, y_limit, xlabel, ylabel):
    ax.plot(x, y, pltcolor, label=pltlabel)
    if x_limit != 0:
        ax.set_xlim(0, x_limit)
    ax.set_ylim(y_min, y_limit)
    ax.legend()
    graph_base_parameter(ax, xlabel, ylabel)

def graph_eu_set(ax, y_min, y_max):
    ax.set_ylim(y_min, y_max)
    ax.set_ylabel('capacity ratio', fontsize= 20, fontweight='bold')
    ax.set_xlabel('cycle', fontsize= 20, fontweight='bold')
    ax.tick_params(direction='in')
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend(prop={"size": 16})

def graph_default(ax, x, y, x_llimit, x_hlimit, x_gap, y_llimit, y_hlimit, y_gap,
                  xlabel, ylabel, lgnd, size, graphcolor,facecolor, graphmarker):
    colors = {0: 'red', 1: 'blue', 2: 'green', 3: 'magenta', 4: 'cyan'}
    if graphcolor in colors:
        ax.scatter(x, y, label=lgnd, s=size, color=colors[graphcolor],
                   edgecolors=colors[graphcolor], facecolors=colors[graphcolor], marker = graphmarker)
    else:
        ax.scatter(x, y, label=lgnd, s=size)
    if x_gap != 0:
        ax.set_xticks(np.arange(x_llimit, x_hlimit, x_gap))
        ax.set_xlim(x_llimit, x_hlimit - x_gap)
    if y_gap != 0:
        ax.set_yticks(np.arange(y_llimit, y_hlimit, y_gap))
        ax.set_ylim(y_llimit, y_hlimit - y_gap)
    graph_base_parameter(ax, xlabel, ylabel)

# Data 엑셀로 output
def output_data(df, sheetname, start_col, start_row, colname, head, use_index = False):
    df.to_excel(writer, sheet_name=sheetname[:30], startcol=start_col, startrow=start_row,
                columns=[colname], header=head, index=use_index)

def output_para_fig(figsaveokchk, filename):
    if figsaveokchk.isChecked():
        if os.path.isfile('d:/'+ filename +'.png'):
            os.remove('d:/'+ filename +'.png')
        fig = plt.gcf()
        fig.savefig('d:/'+ filename +'.png')

# 그래프를 D드라이브에 그림 파일로 저장하는 옵션
def output_fig(figsaveokchk, filename):
    # d 드라이브에 기본 저장
    if figsaveokchk.isChecked():
        if os.path.isfile('d:/'+ filename +'.png'):
            os.remove('d:/'+ filename +'.png')
        plt.savefig('d:/'+ filename +'.png')

# 랜덤한 값 생성 함수
def generate_params(ca_mass_min, ca_mass_max, ca_slip_min, ca_slip_max, an_mass_min, an_mass_max, an_slip_min, an_slip_max):
    ca_mass = np.random.uniform(ca_mass_min, ca_mass_max)
    ca_slip = np.random.uniform(ca_slip_min, ca_slip_max)
    an_mass = np.random.uniform(an_mass_min, an_mass_max)
    an_slip = np.random.uniform(an_slip_min, an_slip_max)
    return ca_mass, ca_slip, an_mass, an_slip

# 전체 결과 기반 dataframe 생성 함수
def generate_simulation_full(ca_ccv_raw, an_ccv_raw, real_raw, ca_mass, ca_slip, an_mass, an_slip,
                             full_cell_max_cap, rated_cap, full_period):
    # 용량 보정
    ca_ccv_raw.ca_cap_new = ca_ccv_raw.ca_cap * ca_mass - ca_slip
    an_ccv_raw.an_cap_new = an_ccv_raw.an_cap * an_mass - an_slip
    # 기준 용량을 x 기준으로 변경
    simul_full_cap = np.arange(0, full_cell_max_cap, 0.1)
    simul_full_ca_volt = np.interp(simul_full_cap, ca_ccv_raw.ca_cap_new, ca_ccv_raw.ca_volt)
    simul_full_an_volt = np.interp(simul_full_cap, an_ccv_raw.an_cap_new, an_ccv_raw.an_volt)
    simul_full_real_volt = np.interp(simul_full_cap, real_raw.real_cap, real_raw.real_volt)
    # 예측되는 full 셀 전압 계산
    simul_full_volt = simul_full_ca_volt - simul_full_an_volt
    # 전체 결과 데이터프레임 생성
    simul_full = pd.DataFrame({"full_cap": simul_full_cap, "an_volt": simul_full_an_volt,
                               "ca_volt": simul_full_ca_volt, "full_volt": simul_full_volt, "real_volt": simul_full_real_volt})
    simul_full = simul_full.drop(simul_full.index[-1])
    # 백분율로 용량 변경
    simul_full.full_cap = simul_full.full_cap / rated_cap * 100
    # 미분값 생성
    simul_full["an_dvdq"] = simul_full.an_volt.diff(periods = full_period) / simul_full.full_cap.diff(periods = full_period)
    simul_full["ca_dvdq"] = simul_full.ca_volt.diff(periods = full_period) / simul_full.full_cap.diff(periods = full_period)
    simul_full["real_dvdq"] = simul_full.real_volt.diff(periods = full_period) / simul_full.full_cap.diff(periods = full_period)
    simul_full["full_dvdq"] = simul_full["ca_dvdq"] - simul_full["an_dvdq"]
    return simul_full

# 토요 데이터 csv 확인/ 폴더, cycle 순으로 입력
def toyo_read_csv(*args): 
    if len(args) == 1:
        filepath = args[0] + "\\capacity.log"
        skiprows = 0
    else:
        filepath = args[0] + "\\%06d" % args[1]
        skiprows = 3
    if os.path.isfile(filepath):
        # read the csv file into a pandas dataframe
        dataraw = pd.read_csv(filepath, sep=",", skiprows=skiprows, engine="c", encoding="cp949", on_bad_lines='skip')
        return dataraw

# Data 처리
# Toyo Profile 불러오기
def toyo_Profile_import(raw_file_path, cycle):
    df = pd.DataFrame()
    df.dataraw = toyo_read_csv(raw_file_path, cycle)
    if hasattr(df, 'dataraw') and not df.dataraw.empty:
        if "PassTime[Sec]" in df.dataraw.columns:
            if "Temp1[Deg]" in df.dataraw.columns:
                # Toyo BLK 3600_3000
                df.dataraw = df.dataraw[["PassTime[Sec]", "Voltage[V]", "Current[mA]", "Condition", "Temp1[Deg]"]]
            else:
            # 신뢰성 충방전기 (Temp 없음)
                df.dataraw = df.dataraw[["PassTime[Sec]", "Voltage[V]", "Current[mA]", "Condition", "TotlCycle"]]
                df.dataraw.columns = ["PassTime[Sec]", "Voltage[V]", "Current[mA]", "Condition", "Temp1[Deg]"]
        else:
            # Toyo BLK5200
            df.dataraw = df.dataraw[["Passed Time[Sec]", "Voltage[V]", "Current[mA]", "Condition", "Temp1[deg]"]]
            df.dataraw.columns = ["PassTime[Sec]", "Voltage[V]", "Current[mA]", "Condition", "Temp1[Deg]"]
    return df

# Toyo Cycle 불러오기

def toyo_cycle_import(raw_file_path):
    df = pd.DataFrame()
    df.dataraw = toyo_read_csv(raw_file_path)
    if hasattr(df, 'dataraw') and not df.dataraw.empty:
        if "Cap[mAh]" in df.dataraw.columns: 
            df.dataraw = df.dataraw[["TotlCycle", "Condition", "Cap[mAh]", "Ocv", "Finish", "Mode", "PeakVolt[V]",
                                     "Pow[mWh]", "PeakTemp[Deg]", "AveVolt[V]"]]
        else: 
            df.dataraw = df.dataraw[["Total Cycle", "Condition", "Capacity[mAh]", "OCV[V]", "End Factor", "Mode",
                                     "Peak Volt.[V]", "Power[mWh]","Peak Temp.[deg]", "Ave. Volt.[V]"]]
            df.dataraw.columns = ["TotlCycle", "Condition", "Cap[mAh]", "Ocv", "Finish", "Mode", "PeakVolt[V]",
                                  "Pow[mWh]", "PeakTemp[Deg]", "AveVolt[V]"]
    return df

# Toyo min. cap 산정하기 (첫 사이클 전류와 C-rate 이용)
def toyo_min_cap(raw_file_path, mincapacity, inirate):
    # 용량 산정 (첫 사이클을 0.2C로 가정)
    if mincapacity == 0:
        if "mAh" in raw_file_path: # 파일 이름에 용량 관련 문자 있을 때
            mincap = name_capacity(raw_file_path)
        else:
            inicapraw = toyo_read_csv(raw_file_path, 1) # 첫 사이클 기준으로 3번째 줄을 C-rate로 가정하여 용량 환산
            mincap = int(round(inicapraw["Current[mA]"].max()/inirate))
    else: # 표기된 숫자를 용량으로 지정
        mincap = mincapacity
    return mincap    

# Toyo Cycle data 처리
def toyo_cycle_data(raw_file_path, mincapacity, inirate, chkir):
    # 폴더 확인
    # print(raw_file_path)
    df = pd.DataFrame()
    # 용량 산정
    tempmincap = toyo_min_cap(raw_file_path, mincapacity, inirate)
    mincapacity = tempmincap
    # data 기본 처리 (csv data loading)
    tempdata = toyo_cycle_import(raw_file_path)
    if hasattr(tempdata, "dataraw") and not tempdata.dataraw.empty:
        Cycleraw = tempdata.dataraw
        # 기존 cycle 저장
        Cycleraw.loc[:,"OriCycle"]=Cycleraw.loc[:,"TotlCycle"]
        # 방전 시작 시 data 변경
        if Cycleraw.loc[0, "Condition"] == 2 and len(Cycleraw.index) > 2:
            if Cycleraw.loc[1, "TotlCycle"] == 1:
                Cycleraw.loc[Cycleraw["Condition"] == 2, "TotlCycle"] -= 1
                Cycleraw = Cycleraw.drop(0, axis=0)
                Cycleraw = Cycleraw.reset_index()
        # Step 충전 용량, 방전 용량, 방전 에너지 계산
        i = 0
        while i < len(Cycleraw) - 1:
                current_cond = Cycleraw.loc[i, "Condition"]
                next_cond = Cycleraw.loc[i + 1, "Condition"]
                if current_cond in (1, 2) and current_cond == next_cond:
                    # 충전/방전 데이터 병합
                    if current_cond == 1:
                        # 충전 데이터 처리
                        Cycleraw.loc[i + 1, "Cap[mAh]"] += Cycleraw.loc[i, "Cap[mAh]"]
                        Cycleraw.loc[i + 1, "Ocv"] = Cycleraw.loc[i, "Ocv"]
                    else:
                        # 방전 데이터 처리
                        Cycleraw.loc[i + 1, "Cap[mAh]"] += Cycleraw.loc[i, "Cap[mAh]"]
                        Cycleraw.loc[i + 1, "Pow[mWh]"] += Cycleraw.loc[i, "Pow[mWh]"]
                        Cycleraw.loc[i + 1, "AveVolt[V]"] = Cycleraw.loc[i + 1, "Pow[mWh]"] / Cycleraw.loc[i + 1, "Cap[mAh]"]
                    # 병합된 행의 사이클 수 감소
                    # Cycleraw.loc[i + 1, "TotlCycle"] -= 1
                    # 현재 행 삭제 및 인덱스 조정
                    Cycleraw = Cycleraw.drop(i, axis=0).reset_index(drop=True)
                    # i += 1  # 병합된 다음 행 건너뛰기
                else:
                    i += 1
        # 충전 용량 처리
        chgdata = Cycleraw[(Cycleraw["Condition"] == 1) & (Cycleraw["Finish"] != "                 Vol") 
                           & (Cycleraw["Finish"] != "Volt") & (Cycleraw["Cap[mAh]"] > (mincapacity/60))]
        chgdata.index = chgdata["TotlCycle"]
        Chg = chgdata["Cap[mAh]"]
        # Rest End Voltage 추출
        Ocv = chgdata["Ocv"]
        # Cycle raw index 변경
        Cycleraw.index = Cycleraw["TotlCycle"]
        # dcir 계산
        dcir = Cycleraw[((Cycleraw["Finish"] == "                 Tim") | (Cycleraw["Finish"] == "Tim") 
                         | (Cycleraw["Finish"] == "Time")) & (Cycleraw["Condition"] == 2) 
                        & (Cycleraw["Cap[mAh]"] < (mincapacity/60))]
        cycnum = dcir["TotlCycle"]
        # 방전 용량/온도 계산
        Dchgdata = Cycleraw[(Cycleraw["Condition"] == 2) & (Cycleraw["Cap[mAh]"] > (mincapacity/60))]
        Dchg = Dchgdata["Cap[mAh]"]
        Temp= Dchgdata["PeakTemp[Deg]"]
        DchgEng = Dchgdata["Pow[mWh]"]
        Chg2 = Chg.shift(periods=-1)
        AvgV = Dchgdata["AveVolt[V]"]
        OriCycle = Dchgdata.loc[:,"OriCycle"]
        # dcir 기본 처리
        for cycle in cycnum:
            if os.path.isfile(raw_file_path + "\\%06d" % cycle):
                dcirpro = pd.read_csv((raw_file_path + "\\%06d" % cycle), sep=",", skiprows=3, engine="c",
                                      encoding="cp949", on_bad_lines='skip')
                if "PassTime[Sec]" in dcirpro.columns:
                    dcirpro = dcirpro[["PassTime[Sec]", "Voltage[V]", "Current[mA]", "Condition", "Temp1[Deg]"]]
                else:
                    dcirpro = dcirpro[["Passed Time[Sec]", "Voltage[V]", "Current[mA]", "Condition", "Temp1[deg]"]]
                    dcirpro.columns = ["PassTime[Sec]", "Voltage[V]", "Current[mA]", "Condition", "Temp1[Deg]"]
                dcircal = dcirpro[(dcirpro["Condition"] == 2)]
                dcir.loc[int(cycle), "dcir"] = ((dcircal["Voltage[V]"].max() - dcircal["Voltage[V]"].min()) 
                                                / round(dcircal["Current[mA]"].max()) * 1000000)
        n = 1
        cyccal = []
        if len(dcir) != 0:
            if (len(Dchg)/(len(dcir)/2)) >= 10:
                dcirstep = (int(len(Dchg)/(len(dcir)/2)/10) + 1) * 10
            else:
                dcirstep = int(len(Dchg)/(len(dcir)/2)) + 1
            for i in range(len(dcir)):
                if chkir:
                    cyccal.append(n)
                    n = n + 1
                else:
                    cyccal.append(n)
                    if i % 2 == 0:
                        n = n + 1
                    else:
                        n = n + dcirstep - 1
        dcir["Cyc"] = cyccal
        dcir = dcir.set_index(dcir["Cyc"])
        # 충방전 효율 계산
        Eff = Dchg/Chg
        Eff2 = Chg2/Dchg
        # 용량 ratio
        Dchg = Dchg/mincapacity
        Chg = Chg/mincapacity
        # 전체 Data 취합
        df.NewData = pd.DataFrame({"Dchg": Dchg, "RndV": Ocv, "Eff": Eff, "Chg": Chg, "DchgEng": DchgEng,
                                   "Eff2": Eff2, "Temp": Temp, "AvgV": AvgV, "OriCyc": OriCycle})
        # df.NewData = df.NewData.dropna(axis=0, how='all')
        df.NewData = df.NewData.dropna(axis=0, how='all', subset=['Dchg'])
        df.NewData = df.NewData.reset_index()
        if hasattr(dcir, "dcir"):
            df.NewData = pd.concat([df.NewData, dcir["dcir"]], axis=1, join="outer")
        else:
            df.NewData.loc[0, "dcir"] = 0
        df.NewData = df.NewData.drop("TotlCycle", axis=1)
    else:
        sys.exit()
    return [mincapacity, df]

# Toyo Step charge Profile data 처리
def toyo_step_Profile_data(raw_file_path, inicycle, mincapacity, cutoff, inirate):
    df = pd.DataFrame()
    # 용량 산정
    tempmincap = toyo_min_cap(raw_file_path, mincapacity, inirate)
    mincapacity = tempmincap
    # data 기본 처리
    if os.path.isfile(raw_file_path + "\\%06d" % inicycle):
        tempdata = toyo_Profile_import(raw_file_path, inicycle)
        stepcyc = inicycle
        lasttime = 0
        if int(tempdata.dataraw["Condition"].max()) < 2:
            df.stepchg = tempdata.dataraw
            df.stepchg = df.stepchg[(df.stepchg["Condition"] == 1)]
            lasttime = df.stepchg["PassTime[Sec]"].max()
            maxcon = 1
            while maxcon == 1:
                stepcyc = stepcyc + 1
                tempdata = toyo_Profile_import(raw_file_path, stepcyc)
                maxcon = int(tempdata.dataraw["Condition"].max())
                tempdata.dataraw = tempdata.dataraw[(tempdata.dataraw["Condition"] == 1)]
                tempdata.dataraw["PassTime[Sec]"] = tempdata.dataraw["PassTime[Sec]"] + lasttime
                df.stepchg = df.stepchg._append(tempdata.dataraw)
                lasttime = df.stepchg["PassTime[Sec]"].max()
        else:
            df.stepchg = tempdata.dataraw
            df.stepchg = df.stepchg[(df.stepchg["Condition"] == 1)]
        if not df.stepchg.empty:
            df.stepchg["Cap[mAh]"] = 0
            # cut-off
            df.stepchg = df.stepchg[df.stepchg["Current[mA]"] >= (cutoff * mincapacity)]
            # 충전 용량 산정
            df.stepchg = df.stepchg.reset_index()
            # 벡터화 연산을 통한 용량 계산
            initial_cap = df.stepchg["Cap[mAh]"].iloc[0]  # 초기 용량 값 추출
            # 1. 시간 차이 계산 (다음 행 - 현재 행)
            df.stepchg["delta_time"] = (
                df.stepchg["PassTime[Sec]"].shift(-1) - df.stepchg["PassTime[Sec]"]
            )
            # 2. 다음 행의 전류값 가져오기
            df.stepchg["next_current"] = df.stepchg["Current[mA]"].shift(-1)
            # 3. 기여도 계산 (시간 차이 * 다음 행 전류 / 3600)
            df.stepchg["contribution"] = (df.stepchg["delta_time"] * df.stepchg["next_current"]) / 3600
            # 4. 누적 합계 계산 및 초기 용량 적용
            df.stepchg["Cap[mAh]"] = initial_cap + df.stepchg["contribution"].fillna(0).cumsum().shift(1, fill_value=0)
            # 5. 임시 컬럼 제거 (선택사항)
            df.stepchg.drop(["delta_time", "next_current", "contribution"], axis=1, inplace=True)
            # 충전 단위 변환
            df.stepchg["PassTime[Sec]"] = df.stepchg["PassTime[Sec]"]/60
            df.stepchg["Current[mA]"] = df.stepchg["Current[mA]"]/mincapacity
            df.stepchg["Cap[mAh]"] = df.stepchg["Cap[mAh]"]/mincapacity
            df.stepchg = df.stepchg[["PassTime[Sec]", "Cap[mAh]", "Voltage[V]", "Current[mA]", "Temp1[Deg]"]]
            df.stepchg.columns = ["TimeMin", "SOC", "Vol", "Crate", "Temp"]
    return [mincapacity, df]

# Toyo 율별 충전 Profile 처리
def toyo_rate_Profile_data(raw_file_path, inicycle, mincapacity, cutoff, inirate):
    df = pd.DataFrame()
    # 용량 산정
    tempmincap = toyo_min_cap(raw_file_path, mincapacity, inirate)
    mincapacity = tempmincap
    # data 기본 처리
    if os.path.isfile(raw_file_path + "\\%06d" % inicycle):
        tempdata = toyo_Profile_import(raw_file_path, inicycle)
        Profileraw0 = tempdata.dataraw
        Profileraw0 = Profileraw0[(Profileraw0["Condition"] == 1)]
        if not Profileraw0.empty:
            df.rateProfile = Profileraw0
            df.rateProfile["Cap[mAh]"] = 0
            # cut-off
            df.rateProfile = df.rateProfile[df.rateProfile["Current[mA]"] >= (cutoff * mincapacity)]
            # 충전 용량 산정
            df.rateProfile = df.rateProfile.reset_index()
            # 벡터화 연산을 통한 용량 계산
            initial_cap = df.rateProfile["Cap[mAh]"].iloc[0]  # 초기 용량 값 추출
            # 1. 시간 차이 계산 (다음 행 - 현재 행)
            df.rateProfile["delta_time"] = (
                df.rateProfile["PassTime[Sec]"].shift(-1) - df.rateProfile["PassTime[Sec]"]
            )
            # 2. 다음 행의 전류값 가져오기
            df.rateProfile["next_current"] = df.rateProfile["Current[mA]"].shift(-1)
            # 3. 기여도 계산 (시간 차이 * 다음 행 전류 / 3600)
            df.rateProfile["contribution"] = (df.rateProfile["delta_time"] * df.rateProfile["next_current"]) / 3600
            # 4. 누적 합계 계산 및 초기 용량 적용
            df.rateProfile["Cap[mAh]"] = initial_cap + df.rateProfile["contribution"].fillna(0).cumsum().shift(1, fill_value=0)
            # 5. 임시 컬럼 제거 (선택사항)
            df.rateProfile.drop(["delta_time", "next_current", "contribution"], axis=1, inplace=True)
            # for i in range(len(df.rateProfile) - 1):
            #     df.rateProfile.loc[i + 1, "Cap[mAh]"] = (df.rateProfile.loc[i + 1, "PassTime[Sec]"] - df.rateProfile.loc[i, "PassTime[Sec]"])/3600 * (df.rateProfile.loc[i + 1, "Current[mA]"]) + df.rateProfile.loc[i, "Cap[mAh]"]
            if len(df.rateProfile) > 1:
                # PassTime[Sec]의 차이 계산 (첫 행은 NaN)
                time_diffs = df.rateProfile["PassTime[Sec]"].diff().iloc[1:]
                # (시간 차이 / 3600) * Current[mA] 계산
                increments = (time_diffs / 3600) * df.rateProfile["Current[mA]"].iloc[1:]
                # 누적 합 계산
                cum_increments = increments.cumsum()
                # 첫 행의 Cap[mAh] 값 가져오기
                initial_cap = df.rateProfile["Cap[mAh]"].iloc[0]
                # 두 번째 행부터 Cap[mAh] 업데이트
                df.rateProfile.iloc[1:, df.rateProfile.columns.get_loc("Cap[mAh]")] = initial_cap + cum_increments.values
            # 충전 단위 변환
            df.rateProfile["PassTime[Sec]"] = df.rateProfile["PassTime[Sec]"]/60
            df.rateProfile["Current[mA]"] = df.rateProfile["Current[mA]"]/mincapacity
            df.rateProfile["Cap[mAh]"] = df.rateProfile["Cap[mAh]"]/mincapacity
            df.rateProfile = df.rateProfile[["PassTime[Sec]", "Cap[mAh]", "Voltage[V]", "Current[mA]", "Temp1[Deg]"]]
            df.rateProfile.columns = ["TimeMin", "SOC", "Vol", "Crate", "Temp"]
    return [mincapacity, df]

# Toyo 충전 Profile 처리
def toyo_chg_Profile_data(raw_file_path, inicycle, mincapacity, cutoff, inirate, smoothdegree):
    df = pd.DataFrame()
    # 용량 산정
    tempmincap = toyo_min_cap(raw_file_path, mincapacity, inirate)
    mincapacity = tempmincap
    # data 기본 처리
    if os.path.isfile(raw_file_path + "\\%06d" % inicycle):
        tempdata = toyo_Profile_import(raw_file_path, inicycle)
        df.Profile = tempdata.dataraw
        # 충전/방전만 추출
        df.Profile = df.Profile[(df.Profile["Condition"] == 1)]
        # cut-off
        df.Profile = df.Profile[df.Profile["Voltage[V]"] >= cutoff]
        if not df.Profile.empty:
            # 방전 용량 산정, dQdV 산정
            df.Profile = df.Profile.reset_index()
            df.Profile["deltime"] = df.Profile["PassTime[Sec]"].diff()
            df.Profile["delcurr"] = df.Profile["Current[mA]"].rolling(window=2).mean()
            df.Profile["delvol"] = df.Profile["Voltage[V]"].rolling(window=2).mean()
            df.Profile["delcap"] = df.Profile["deltime"]/3600 * df.Profile["delcurr"]/mincapacity
            df.Profile["delwh"] = df.Profile["delcap"] * mincapacity * df.Profile["delvol"]
            df.Profile["Cap[mAh]"] = df.Profile["delcap"].cumsum()
            df.Profile["Chgwh"] = df.Profile["delwh"].cumsum()
            if smoothdegree == 0:
                smoothdegree = len(df.Profile) / 30
            df.Profile["delvol"] = df.Profile["Voltage[V]"].diff(periods=smoothdegree)
            df.Profile["delcap"] = df.Profile["Cap[mAh]"].diff(periods=smoothdegree)
            df.Profile["dQdV"] = df.Profile["delcap"]/df.Profile["delvol"]
            df.Profile["dVdQ"] = df.Profile["delvol"]/df.Profile["delcap"]
            # 방전 단위 변환
            df.Profile["PassTime[Sec]"] = df.Profile["PassTime[Sec]"]/60
            df.Profile["Current[mA]"] = df.Profile["Current[mA]"]/mincapacity
            df.Profile = df.Profile[["PassTime[Sec]", "Cap[mAh]", "Chgwh", "Voltage[V]", "Current[mA]",
                                     "dQdV", "dVdQ", "Temp1[Deg]"]]
            df.Profile.columns = ["TimeMin", "SOC", "Energy", "Vol", "Crate", "dQdV", "dVdQ", "Temp"]
    return [mincapacity, df]

# Toyo 방전 Profile 처리
def toyo_dchg_Profile_data(raw_file_path, inicycle, mincapacity, cutoff, inirate, smoothdegree):
    df = pd.DataFrame()
    # 용량 산정
    tempmincap = toyo_min_cap(raw_file_path, mincapacity, inirate)
    mincapacity = tempmincap
    # data 기본 처리
    if os.path.isfile(raw_file_path + "\\%06d" % inicycle):
        tempdata = toyo_Profile_import(raw_file_path, inicycle)
        df.Profile = tempdata.dataraw
        df.Profile = df.Profile[(df.Profile["Condition"] == 2)]
        # 뒷 사이클 있는지 확인
        if os.path.isfile(raw_file_path + "\\%06d" % (inicycle + 1)):
            tempdata2 = toyo_Profile_import(raw_file_path, inicycle + 1)
            df.Profile2 = tempdata2.dataraw
            # 뒷 사이클에 충전이 있는지 확인
            if not tempdata2.dataraw["Condition"].isin([1]).any():
                # 방전만 추출하여 기존 데이터의 시간을 합하고 기존 df에 추가
                lasttime = df.Profile["PassTime[Sec]"].max()
                df.Profile2 = df.Profile2[(df.Profile2["Condition"] == 2)]
                df.Profile2["PassTime[Sec]"] = df.Profile2["PassTime[Sec]"] + lasttime
                df.Profile = df.Profile._append(df.Profile2)
        # cut-off
        df.Profile = df.Profile[df.Profile["Voltage[V]"] >= cutoff]
        if not df.Profile.empty:
            # 방전 용량 산정, dQdV 산정
            df.Profile = df.Profile.reset_index()
            df.Profile["deltime"] = df.Profile["PassTime[Sec]"].diff()
            df.Profile["delcurr"] = df.Profile["Current[mA]"].rolling(window=2).mean()
            df.Profile["delvol"] = df.Profile["Voltage[V]"].rolling(window=2).mean()
            df.Profile["delcap"] = df.Profile["deltime"]/3600 * df.Profile["delcurr"]/mincapacity
            df.Profile["delwh"] = df.Profile["delcap"] * mincapacity * df.Profile["delvol"]
            df.Profile["Cap[mAh]"] = df.Profile["delcap"].cumsum()
            df.Profile["Dchgwh"] = df.Profile["delwh"].cumsum()
            if smoothdegree == 0:
                smoothdegree = len(df.Profile) / 30
            df.Profile["delvol"] = df.Profile["Voltage[V]"].diff(periods=smoothdegree)
            df.Profile["delcap"] = df.Profile["Cap[mAh]"].diff(periods=smoothdegree)
            df.Profile["dQdV"] = df.Profile["delcap"]/df.Profile["delvol"]
            df.Profile["dVdQ"] = df.Profile["delvol"]/df.Profile["delcap"]
            # 방전 단위 변환
            df.Profile["PassTime[Sec]"] = df.Profile["PassTime[Sec]"]/60
            df.Profile["Current[mA]"] = df.Profile["Current[mA]"]/mincapacity
            df.Profile = df.Profile[["PassTime[Sec]", "Cap[mAh]", "Dchgwh", "Voltage[V]", "Current[mA]",
                                     "dQdV", "dVdQ", "Temp1[Deg]"]]
            df.Profile.columns = ["TimeMin", "SOC", "Energy", "Vol", "Crate", "dQdV", "dVdQ", "Temp"]
    return [mincapacity, df]

# Toyo Step charge Profile data 처리
def toyo_Profile_continue_data(raw_file_path, inicycle, endcycle, mincapacity, inirate):
    df = pd.DataFrame()
    # 용량 산정
    tempmincap = toyo_min_cap(raw_file_path, mincapacity, inirate)
    mincapacity = tempmincap
    # data 기본 처리
    if os.path.isfile(raw_file_path + "\\%06d" % inicycle):
        tempdata = toyo_Profile_import(raw_file_path, inicycle)
        stepcyc = inicycle
        lasttime = 0
        if int(tempdata.dataraw["Condition"].max()) < 2:
            df.stepchg = tempdata.dataraw
            lasttime = df.stepchg["PassTime[Sec]"].max()
            maxcon = 1
            while maxcon == 1:
                stepcyc = stepcyc + 1
                tempdata = toyo_Profile_import(raw_file_path, stepcyc)
                maxcon = int(tempdata.dataraw["Condition"].max())
                tempdata.dataraw["PassTime[Sec]"] = tempdata.dataraw["PassTime[Sec]"] + lasttime
                df.stepchg = df.stepchg._append(tempdata.dataraw)
                lasttime = df.stepchg["PassTime[Sec]"].max()
        else:
            df.stepchg = tempdata.dataraw
        if not df.stepchg.empty:
            df.stepchg["Cap[mAh]"] = 0
            # 충전 용량 산정
            df.stepchg = df.stepchg.reset_index()
            # for i in range(len(df.stepchg) - 1):
            #     df.stepchg.loc[i + 1, "Cap[mAh]"] = (df.stepchg.loc[i + 1, "PassTime[Sec]"] - df.stepchg.loc[i, "PassTime[Sec]"])/3600 * (df.stepchg.loc[i + 1, "Current[mA]"]) + df.stepchg.loc[i, "Cap[mAh]"]
            if len(df.stepchg) > 1:
                # PassTime[Sec]의 차이 계산 (첫 행은 NaN)
                time_diffs = df.stepchg["PassTime[Sec]"].diff().iloc[1:]
                # (시간 차이 / 3600) * Current[mA] 계산
                increments = (time_diffs / 3600) * df.stepchg["Current[mA]"].iloc[1:]
                # 누적 합 계산
                cum_increments = increments.cumsum()
                # 첫 행의 Cap[mAh] 값 가져오기
                initial_cap = df.stepchg["Cap[mAh]"].iloc[0]
                # 두 번째 행부터 Cap[mAh] 업데이트
                df.stepchg.iloc[1:, df.stepchg.columns.get_loc("Cap[mAh]")] = initial_cap + cum_increments.values
            # 충전 단위 변환
            df.stepchg["PassTime[Sec]"] = df.stepchg["PassTime[Sec]"]/60
            df.stepchg["Current[mA]"] = df.stepchg["Current[mA]"]/mincapacity
            df.stepchg["Cap[mAh]"] = df.stepchg["Cap[mAh]"]/mincapacity
            df.stepchg = df.stepchg[["PassTime[Sec]", "Cap[mAh]", "Voltage[V]", "Current[mA]", "Temp1[Deg]"]]
            df.stepchg.columns = ["TimeMin", "SOC", "Vol", "Crate", "Temp"]
    return [mincapacity, df]

# PNE Profile data 기본 input 처리
def pne_data(raw_file_path, inicycle):
    df = pd.DataFrame()
    if os.path.isdir(raw_file_path + "\\Restore\\"):
        rawdir = raw_file_path + "\\Restore\\"
        # Profile에 사용할 파일 선정
        filepos = pne_search_cycle(rawdir, inicycle, inicycle + 1)
        # for files in subfile:
        if os.path.isdir(rawdir) and (filepos[0] != -1):
            subfile = [f for f in os.listdir(rawdir) if f.endswith(".csv")]
            for files in subfile[(filepos[0]):(filepos[1] + 1)]:
                # SaveData가 있는 파일을 순서대로 확인하면 Profile 작성
                if "SaveData" in files:
                    df.Profilerawtemp = pd.read_csv( (rawdir + files), sep=",", skiprows=0, engine="c", header=None,
                                                    encoding="cp949", on_bad_lines='skip')
                    if hasattr(df, "Profileraw"):
                        df.Profileraw = pd.concat([df.Profileraw, df.Profilerawtemp], ignore_index=True)
                    else:
                        df.Profileraw = df.Profilerawtemp 
    return df

# PNE에서 원하는 사이클이 들어있는 파일명을 찾는 코드
def pne_search_cycle(rawdir, start, end):
    # Profile에 사용할 파일 선정
    if os.path.isdir(rawdir):
        subfile = [f for f in os.listdir(rawdir) if f.endswith(".csv")]
        for files in subfile:
            # SaveEndData가 있는 파일 확인
            if "SaveEndData" in files:
                df = pd.read_csv(rawdir + files, sep=",", skiprows=0, engine="c", header=None, encoding="cp949",
                                 on_bad_lines='skip')
                if start != 1:
                    index_min = df.loc[(df.loc[:,27] == (start - 1)), 0].tolist()
                else:
                    index_min = [0]
                index_max = df.loc[(df.loc[:,27] == end), 0].tolist()
                if not index_max:
                    index_max = df.loc[(df.loc[:,27] == df.loc[:,27].max()), 0].tolist()
                df2 = pd.read_csv(rawdir + "savingFileIndex_start.csv", delim_whitespace=True, skiprows=0, engine="c",
                                  header=None, encoding="cp949", on_bad_lines='skip')
                df2 = df2.loc[:,3].tolist()
                index2 = []
                for element in df2:
                    new_element = int(element.replace(',', ''))
                    index2.append(new_element)
                if len(index_min) != 0:
                    file_start = binary_search(index2, index_min[-1] + 1) - 1
                    file_end = binary_search(index2, index_max[-1]) - 1
                else:
                    file_start = -1
                    file_end = -1
    return [file_start, file_end]

# 연속된 데이터의 Profile을 찾아서 확인
def pne_continue_data(raw_file_path, inicycle, endcycle):
    df = pd.DataFrame()
    if os.path.isdir(raw_file_path + "\\Restore\\"):
        rawdir = raw_file_path + "\\Restore\\"
        # Profile에 사용할 파일 선정
        if os.path.isdir(rawdir):
            subfile = [f for f in os.listdir(rawdir) if f.endswith(".csv")]
            filepos = pne_search_cycle(rawdir, inicycle, endcycle)
            # for files in subfile:
            if filepos[0] != -1:
                for files in subfile[(filepos[0]):(filepos[1] + 1)]:
                    # SaveData가 있는 파일을 순서대로 확인하면 Profile 작성
                    if "SaveData" in files:
                        df.Profilerawtemp = pd.read_csv( (rawdir + files), sep=",", skiprows=0, engine="c",
                                                        header=None, encoding="cp949", on_bad_lines='skip')
                        if hasattr(df, "Profileraw"):
                            df.Profileraw = pd.concat([df.Profileraw, df.Profilerawtemp], ignore_index=True)
                        else:
                            df.Profileraw = df.Profilerawtemp 
            elif filepos[0] == -1 and inicycle == 1:
                for files in subfile[0:(filepos[1] + 1)]:
                    # SaveData가 있는 파일을 순서대로 확인하면 Profile 작성
                    if "SaveData" in files:
                        df.Profilerawtemp = pd.read_csv( (rawdir + files), sep=",", skiprows=0, engine="c",
                                                        header=None, encoding="cp949", on_bad_lines='skip')
                        if hasattr(df, "Profileraw"):
                            df.Profileraw = pd.concat([df.Profileraw, df.Profilerawtemp], ignore_index=True)
                        else:
                            df.Profileraw = df.Profilerawtemp 
    return df

def pne_cyc_continue_data(raw_file_path):
    df = pd.DataFrame()
    if os.path.isdir(raw_file_path + "\\Restore\\"):
        rawdir = raw_file_path + "\\Restore\\"
        # Profile에 사용할 파일 선정
        if os.path.isdir(rawdir):
            subfile = [f for f in os.listdir(rawdir) if f.endswith(".csv")]
            # for files in subfile:
            for files in subfile:
                # SaveData가 있는 파일을 순서대로 확인하면 Profile 작성
                if "SaveEndData" in files:
                    df.Cycrawtemp = pd.read_csv((rawdir + files), sep=",", skiprows=0, engine="c",
                                                header=None, encoding="cp949", on_bad_lines='skip')
    return df

# PNE channel No., mincapacity 산정 기본 처리
def pne_min_cap(raw_file_path, mincapacity, ini_crate):
    # 용량 산정
    if mincapacity == 0:
        if "mAh" in raw_file_path:
            mincapacity = name_capacity(raw_file_path)
        elif os.path.isdir(raw_file_path + "\\Restore\\"):
            subfile = [f for f in os.listdir(raw_file_path + "\\Restore\\") if f.endswith('.csv')]
            for files in subfile:
                if ("SaveData0001.csv" in files):
                    if os.stat(raw_file_path + "\\Restore\\" + files).st_size != 0:
                        inicapraw = pd.read_csv(raw_file_path + "\\Restore\\" + files, sep=",", skiprows=0, engine="c",
                                                header=None, encoding="cp949", on_bad_lines='skip')
                        if len(inicapraw) > 2:
                            mincapacity = int(round(abs(inicapraw.iloc[2, 9]/1000))/ini_crate)
    return mincapacity

# PNE Cycle data 처리
def pne_simul_cycle_data(raw_file_path, min_capacity, ini_crate):
    '''0:Index 1: 2:StepType(1:충전,2:방전,3:휴지,8:loop) 3:ChgDchg 4: 5:충전
    6:EndState(64:휴지,64:loop,65:전압,66:전류-충전,78:용량) 7:Step 8:Voltage(mV) 9:Current(A) 10:Chg Capacity(mAh)
    11:Dchg Capacity(mAh) 12:Chg Power(W) 13:Dchg Power(W) 14:Chg WattHour(Wh) 15:Dchg WattHour(Wh) 
    16: 17:StepTime(s) 18: 19: 20:imp 
    21: 22: 23: 24:Temperature(°C) 25: 26: 27:Total Cycle 28:CurrCycle 29:Average voltage(mV) 30:Average current(A) 
    31: 32: 33:day 34:time 35: 36: 37: 38: 39: 40: 
    41: 42: 43: 44: 45:voltage max 46: '''
    df_all = pd.DataFrame()
    df02 = pd.DataFrame()
    df02_cap_max = 0
    df05 = pd.DataFrame()
    df05_cap_max = 0
    df05_long_cycle = []
    df05_long_value = []
    if (raw_file_path[-4:-1]) != "ter":
        # PNE 채널, 용량 산정
        mincapacity = pne_min_cap(raw_file_path, min_capacity, ini_crate)
        # data 기본 처리 (csv data loading)
        if os.path.isdir(raw_file_path + "\\Restore\\"):
            subfile = [f for f in os.listdir(raw_file_path + "\\Restore\\") if f.endswith('.csv')]
            for files in subfile:
                if "SaveEndData.csv" in files:
                    if os.stat(raw_file_path + "\\Restore\\" + files).st_size != 0:
                        Cycleraw = pd.read_csv(raw_file_path + "\\Restore\\" + files, sep=",", skiprows=0, engine="c",
                                               header=None, encoding="cp949", on_bad_lines='skip')
                        Cycleraw = Cycleraw[[27, 2, 11, 9, 24, 6, 8]]
                        Cycleraw.columns = ["TotlCycle", "Condition", "DchgCap", "Curr", "Temp", "EndState", "Vol"]
        # Cycleraw["OriCycle"] = Cycleraw["TotlCycle"]
        # condition을 기준으로 8을 대표로 해서 용량 산정
        # 전류의 경우 min을 기준으로 산정
        # 전압의 경우 충전 max 전압을 기준으로 산정
        max_cap = (Cycleraw.query("Condition == 2")).pivot_table(index="TotlCycle", columns="Condition",
                                                                 values = "DchgCap", aggfunc = "sum")
        max_vol = (Cycleraw.query("Condition == 1")).pivot_table(index="TotlCycle", columns="Condition",
                                                                 values = "Vol", aggfunc = "max")
        min_vol = (Cycleraw.query("Condition == 2")).pivot_table(index="TotlCycle", columns="Condition",
                                                                 values = "Vol", aggfunc = "min")
        min_crate = (Cycleraw.query("Condition == 2")).pivot_table(index="TotlCycle", columns="Condition",
                                                                   values = "Curr", aggfunc = "max")
        avg_temp = (Cycleraw.query("Condition == 2")).pivot_table(index="TotlCycle", columns="Condition",
                                                                  values = "Temp", aggfunc = "mean")
        df_all = pd.DataFrame({"Temp": avg_temp.iloc[:,0], "Curr": min_crate.iloc[:,0], "Dchg": max_cap.iloc[:, 0],
                               "max_vol": max_vol.iloc[:, 0], "min_vol": min_vol.iloc[:, 0]})
        df_all["Temp"] = df_all["Temp"]/1000
        df_all["Curr"] = - 1 * df_all["Curr"]/mincapacity/1000
        df_all["max_vol"] = df_all["max_vol"]/1000
        df_all["Dchg"] = df_all["Dchg"]/mincapacity/1000
        df_all["min_vol"] = df_all["min_vol"]/1000
        df05 = df_all.query('0.490 < Curr < 0.510')
        j = 0
        if len(df05) > 40:
            df05["Dchg_Diff"] = df05["Dchg"].diff()
            df05["max_vol_diff"] = df05["max_vol"].diff()
            df05["min_vol_diff"] = df05["min_vol"].diff()
            df05 = df05.loc[df05["Dchg"].idxmax():]
            df05_cap_max = df05["Dchg"].iloc[0] - df05["Dchg_Diff"].iloc[0:30].mean() * float(df05.index[0])
            df05["Dchg"] = df05["Dchg"] / df05_cap_max
            df05["long"] = 0
        # 장수명 부분 제거 관련 코드
            for i in range(len(df05) - 1):
                if((df05["max_vol_diff"].iloc[i] < -15) | (df05["min_vol_diff"].iloc[i] > 50)) & ( i > 0):
                    df05["long"].iloc[i] =  df05["Dchg_Diff"].iloc[i]
                    df05_long_cycle.append(df05.index[i])
                    df05_long_value.append(df05["Dchg_Diff"].iloc[i])
            df05["long_acc"] = df05["long"].cumsum()
        df02 = df_all.query('0.190 < Curr < 0.210')
        df02_max_vol = df_all["max_vol"].max()
        df02 = df02[df02["max_vol"] > (df02_max_vol - 10)]
        if len(df02) > 3:
            df02 = df02.iloc[1:]
            if (df02.index[1] - df02.index[0]) < 40:
                df02 = df02.iloc[1::2]
            df02.index = df02.index - df02.index[0]
            df02["Dchg_Diff"] = df02["Dchg"].diff()
            df02 = df02.loc[df02["Dchg"].idxmax():]
            df02_cap_max = df02["Dchg"].max() - df02["Dchg_Diff"].iloc[1] * df02.index[0] / (df02.index[1] - df02.index[0])
            df02["Dchg"] = df02["Dchg"] / df02_cap_max
    return [mincapacity, df05, df05_cap_max, df02, df02_cap_max, df05_long_cycle, df05_long_value, df_all]

# PNE Cycle data 처리
def pne_simul_cycle_data_file(df_all, raw_file_path, min_capacity, ini_crate):
    '''0:Index 1: 2:StepType(1:충전,2:방전,3:휴지,8:loop) 3:ChgDchg 4: 5:충전
    6:EndState(64:휴지,64:loop,65:전압,66:전류-충전,78:용량) 7:Step 8:Voltage(mV) 9:Current(A) 10:Chg Capacity(mAh)
    11:Dchg Capacity(mAh) 12:Chg Power(W) 13:Dchg Power(W) 14:Chg WattHour(Wh) 15:Dchg WattHour(Wh) 
    16: 17:StepTime(s) 18: 19: 20:imp 
    21: 22: 23: 24:Temperature(°C) 25: 26: 27:Total Cycle 28:CurrCycle 29:Average voltage(mV) 30:Average current(A) 
    31: 32: 33:day 34:time 35: 36: 37: 38: 39: 40: 
    41: 42: 43: 44: 45:voltage max 46: '''
    # df_all = pd.DataFrame()
    df02 = pd.DataFrame()
    df02_cap_max = 0
    df05 = pd.DataFrame()
    df05_cap_max = 0
    df05_long_cycle = []
    df05_long_value = []
    # PNE 채널, 용량 산정
    mincapacity = pne_min_cap(raw_file_path, min_capacity, ini_crate)
    # data 기본 처리 (csv data loading)
    df05 = df_all.query('0.490 < Curr < 0.510')
    j = 0
    if len(df05) > 40:
        df05["Dchg_Diff"] = df05["Dchg"].diff()
        df05["max_vol_diff"] = df05["max_vol"].diff()
        df05["min_vol_diff"] = df05["min_vol"].diff()
        df05 = df05.loc[df05["Dchg"].idxmax():]
        df05_cap_max = df05["Dchg"].iloc[0] - df05["Dchg_Diff"].iloc[0:30].mean() * float(df05.index[0])
        df05["Dchg"] = df05["Dchg"] / df05_cap_max
        df05["long"] = 0
    # 장수명 부분 제거 관련 코드
        for i in range(len(df05) - 1):
            if((df05["max_vol_diff"].iloc[i] < -15) | (df05["min_vol_diff"].iloc[i] > 50)) & ( i > 0):
                df05["long"].iloc[i] =  df05["Dchg_Diff"].iloc[i]
                df05_long_cycle.append(df05.index[i])
                df05_long_value.append(df05["Dchg_Diff"].iloc[i])
        df05["long_acc"] = df05["long"].cumsum()
    df02 = df_all.query('0.190 < Curr < 0.210')
    df02_max_vol = df_all["max_vol"].max()
    df02 = df02[df02["max_vol"] > (df02_max_vol - 10)]
    if len(df02) > 3:
        df02 = df02.iloc[1:]
        if (df02.index[1] - df02.index[0]) < 40:
            df02 = df02.iloc[1::2]
        df02.index = df02.index - df02.index[0]
        df02["Dchg_Diff"] = df02["Dchg"].diff()
        df02 = df02.loc[df02["Dchg"].idxmax():]
        df02_cap_max = df02["Dchg"].max() - df02["Dchg_Diff"].iloc[1] * df02.index[0] / (df02.index[1] - df02.index[0])
        df02["Dchg"] = df02["Dchg"] / df02_cap_max
    return [mincapacity, df05, df05_cap_max, df02, df02_cap_max, df05_long_cycle, df05_long_value, df_all]

# PNE Cycle data 처리
def pne_cycle_data(raw_file_path, mincapacity, ini_crate, chkir, chkir2, mkdcir):
    '''0/Index/1/-/2/StepType(1_충전,2_방전,3_휴지,8_loop)/3/ChgDchg/4/-/5/충전/6/EndState(66_충전,65_방전,64_휴지,64_loop)/
    7/Step/8/Voltage(mV)/9/Current(A)/10/ChgCapacity(mAh)/11/DchgCapacity(mAh)/12/ChgPower(W)/13/DchgPower(W)/
    14/ChgWattHour(Wh)/ 15/DchgWattHour(Wh)/16/-/17/StepTime(s)/18/-/19/-/20/imp/21/-/22/-/23/-/
    24/Temperature(°C)/25/-/26/-/27/TotalCycle/28/CurrCycle/29/AverageVoltage(mV)/30/AverageCurrent(A)/
    31/-/32/-/33/day/34/time/35/-/36/-/37/-/38/-/39/-/40/-/41/-/42/-/43/-/44/-/45/voltage_max/46/-'''
    # 폴더 확인
    # print(raw_file_path)
    df = pd.DataFrame()
    if (raw_file_path[-4:-1]) != "ter":
        # PNE 채널, 용량 산정
        mincapacity = pne_min_cap(raw_file_path, mincapacity, ini_crate)
        # data 기본 처리 (csv data loading)
        if os.path.isdir(raw_file_path + "\\Restore\\"):
            subfile = [f for f in os.listdir(raw_file_path + "\\Restore\\") if f.endswith('.csv')]
            for files in subfile:
                if "SaveEndData.csv" in files:
                    if os.stat(raw_file_path + "\\Restore\\" + files).st_size > 0 and mincapacity is not None:
                        Cycleraw = pd.read_csv(raw_file_path + "\\Restore\\" + files, sep=",", skiprows=0, engine="c",
                                               header=None, encoding="cp949", on_bad_lines='skip')
                        Cycleraw = Cycleraw[[27, 2, 10, 11, 8, 20, 45, 15, 17, 9, 24, 29, 6]]
                        Cycleraw.columns = ["TotlCycle", "Condition", "chgCap", "DchgCap", "Ocv", "imp", "volmax",
                                            "DchgEngD", "steptime", "Curr", "Temp", "AvgV", "EndState"]
                        # PNE 기본 DCIR (연속 기준 10s pulse, 10s 이내 시간의 경우 단순 pulse 기준 끝나는 시간 기준)
                        if ('PNE21' in raw_file_path) or ('PNE22' in raw_file_path):
                            Cycleraw.DchgCap = Cycleraw.DchgCap/1000
                            Cycleraw.chgCap = Cycleraw.chgCap/1000
                            Cycleraw.Curr = Cycleraw.Curr/1000
                        if chkir:
                            dcirtemp = Cycleraw[(Cycleraw["Condition"] == 2) & (Cycleraw["volmax"] > 4100000)]
                            dcirtemp.index = dcirtemp["TotlCycle"]
                            dcir = dcirtemp.imp/1000
                            dcir = dcir[~dcir.index.duplicated()]
                        # 1s pulse, RSS DCIR
                        elif mkdcir:
                            # dcirtemp1 - SOC 종료로 끝나는 Step 중에 0.15C 이상 충방전 C-rate 기준으로 선정 (RSS CCV)
                            dcirtemp1 = Cycleraw.loc[(Cycleraw['EndState'] == 78) 
                                                     & (Cycleraw['Curr'].abs() >= 0.15 * mincapacity * 1000)]
                            # dcirtemp2 - 1s 충방전 DCIR을 위한 pulse CCV 선정
                            dcirtemp2 = Cycleraw.loc[(Cycleraw["steptime"] == 100) 
                                                     & (Cycleraw["EndState"] == 64)
                                                     & (Cycleraw["Condition"].isin([1, 2]))]
                            # dcirtemp3 - pulse 후 rest 전압 산정 (RSS OCV)
                            dcirtemp3 = Cycleraw.loc[(Cycleraw['steptime'].isin([90000, 180000, 186000, 546000])) 
                                                     & (Cycleraw['EndState'] == 64) 
                                                     & (Cycleraw['Condition'] == 3)]
                            # dcri 계산 - dcirtemp2.imp - 1s pulse, dcirtemp1.imp - RSS
                            min_dcir_count = min(len(dcirtemp1), len(dcirtemp2), len(dcirtemp3))
                            dcirtemp1 = dcirtemp1.iloc[:min_dcir_count]
                            dcirtemp2 = dcirtemp2.iloc[:min_dcir_count]
                            dcirtemp3 = dcirtemp3.iloc[:min_dcir_count]
                            if (len(dcirtemp3) != 0) and (len(dcirtemp1) != 0) and (len(dcirtemp2) != 0):
                                for i in range(0, min_dcir_count):
                                    current1 = dcirtemp1.iloc[i, 9]
                                    current2 = dcirtemp2.iloc[i, 9]
                                    if current1 != 0 and (current1 - current2) != 0:  # 0으로 나누는 것 방지
                                        dcirtemp1.iloc[i, 5] = abs((dcirtemp3.iloc[i, 4] - dcirtemp1.iloc[i, 4]) / current1 * 1000)
                                        dcirtemp2.iloc[i, 5] = abs((dcirtemp2.iloc[i, 4] - dcirtemp1.iloc[i, 4]) / (current1 - current2) * 1000)
                                    else:
                                        dcirtemp1.iloc[i, 5] = None  # 또는 다른 기본값 설정
                                        dcirtemp2.iloc[i, 5] = None
                        # SOC5,50 10s pulse DCIR
                        else:
                            # pulse 기준, 1분 이하 pulse 기준으로 산정
                            dcirtemp = Cycleraw[(Cycleraw["Condition"] == 2) & (Cycleraw["steptime"] <= 6000)]
                            dcirtemp["dcir"] = dcirtemp.imp/1000
                        # 필요한 모든 값을 한 번에 계산
                        pivot_data = Cycleraw.pivot_table(
                            index="TotlCycle",
                            columns="Condition",
                            values=["DchgCap", "DchgEngD", "chgCap", "Ocv", "Temp"],
                            aggfunc={
                                "DchgCap": "sum",
                                "DchgEngD": "sum",
                                "chgCap": "sum",
                                "Ocv": "min",
                                "Temp": "max"
                            }
                        )
                        # 각 계산 결과를 추출
                        Dchg = pivot_data["DchgCap"][2] / mincapacity / 1000
                        DchgEng = pivot_data["DchgEngD"][2] / 1000
                        Chg = pivot_data["chgCap"][1] / mincapacity / 1000
                        Ocv = pivot_data["Ocv"][3] / 1000000
                        Temp = pivot_data["Temp"][2] / 1000
                        # ChgCap2 계산
                        ChgCap2 = Chg.shift(periods=-1)
                        # Eff, Eff2, AvgV 계산
                        Eff = Dchg / Chg
                        Eff2 = ChgCap2 / Dchg
                        AvgV = DchgEng / Dchg / mincapacity * 1000
                        # OriCycle 생성
                        OriCycle = pd.Series(Dchg.index)
                        if chkir and len(OriCycle) == len(dcir):
                            df.NewData = pd.concat([Dchg, Ocv, Eff, Chg, DchgEng, Eff2, dcir, Temp, AvgV, OriCycle], axis=1).reset_index(drop=True)
                            df.NewData.columns = ["Dchg", "RndV", "Eff", "Chg", "DchgEng", "Eff2", "dcir", "Temp", "AvgV", "OriCyc"]
                        if chkir:
                            df.NewData = pd.concat([Dchg, Ocv, Eff, Chg, DchgEng, Eff2, Temp, AvgV, OriCycle], axis=1).reset_index(drop=True)
                            df.NewData.columns = ["Dchg", "RndV", "Eff", "Chg", "DchgEng", "Eff2", "Temp", "AvgV", "OriCyc"]
                            df.NewData.loc[0, "dcir"] = 0
                        elif mkdcir and (len(dcirtemp3) != 0) and (len(dcirtemp1) != 0):
                            if chkir2:
                                cyccal = range(1, len(dcirtemp1)+1)
                            # dcir - RSS, dcir2 - 1s pulse 
                            if (len(dcirtemp1) != 0) and (len(dcirtemp2) != 0):
                                dcirtemp1 = same_add(dcirtemp1, "TotlCycle")
                                dcir = pd.DataFrame({"Cyc": dcirtemp1["TotlCycle_add"], "dcir_raw2": dcirtemp1["imp"]})
                                dcir = dcir.set_index(dcir["Cyc"])
                                dcirtemp2 = same_add(dcirtemp2, "TotlCycle")
                                dcir2 = pd.DataFrame({"Cyc": dcirtemp2["TotlCycle_add"], "dcir_raw": dcirtemp2["imp"]})
                                dcir2 = dcir2.set_index(dcir2["Cyc"])
                                dcirtemp3 = same_add(dcirtemp3, "TotlCycle")
                                df_rssocv = pd.DataFrame({"Cyc": dcirtemp3["TotlCycle_add"], "rssocv": dcirtemp3["Ocv"]/1000000})
                                df_rssocv = df_rssocv.set_index(dcir["Cyc"])
                                df_rssccv = pd.DataFrame({"Cyc": dcirtemp1["TotlCycle_add"], "rssccv": dcirtemp1["Ocv"]/1000000})
                                df_rssccv = df_rssccv.set_index(dcir["Cyc"])
                                df.NewData = pd.concat([Dchg, Ocv, Eff, Chg, DchgEng, Eff2, Temp, AvgV, OriCycle], axis=1).reset_index(drop=True)
                                df.NewData.columns = ["Dchg", "RndV", "Eff", "Chg", "DchgEng", "Eff2", "Temp", "AvgV", "OriCyc"]
                                if hasattr(dcir2, "dcir_raw"):
                                    df.NewData["dcir"] = dcir["dcir_raw2"]
                                    df.NewData["dcir2"] = dcir2["dcir_raw"]
                                    df.NewData["rssocv"] = df_rssocv["rssocv"]
                                    df.NewData["rssccv"] = df_rssccv["rssccv"]
                                else:
                                    df.NewData.loc[0, "dcir"] = 0
                                    df.NewData.loc[0, "dcir2"] = 0
                                    df.NewData.loc[0, "rssocv"] = 0
                                    df.NewData.loc[0, "rssccv"] = 0
                                soc70_dcir = df.NewData.dcir2.dropna(axis=0)
                                soc70_rss_dcir = df.NewData.dcir.dropna(axis=0)
                                # SOC70의 데이터만 그래프 표기
                                if (len(soc70_dcir) // 6)  > (len(df.NewData.index) // 100):
                                    # 6개 중에 4번째 것만 추출
                                    soc70_dcir = soc70_dcir[3:][::6]
                                    soc70_rss_dcir = soc70_rss_dcir[3:][::6]
                                else:
                                    # 4개 중에 1번째 것만 추출
                                    soc70_dcir = soc70_dcir[::4]
                                    soc70_rss_dcir = soc70_rss_dcir[::4]
                                df.NewData["soc70_dcir"] = soc70_dcir
                                df.NewData["soc70_rss_dcir"] = soc70_rss_dcir
                        else:
                            if ('dcirtemp' in locals()):
                                if not chkir2:
                                    n = 1
                                    cyccal = []
                                    if len(dcirtemp) != 0:
                                        dcirstep = max(1, int(len(Dchg) / len(dcirtemp) * 2 / 10) * 10)
                                        for i in range(len(dcirtemp)):
                                            cyccal.append(n)
                                            n += 1 if i % 2 == 0 else dcirstep - 1
                                else:
                                    cyccal = range(1, len(dcirtemp)+1)
                                if hasattr(dcirtemp, "dcir") and not dcirtemp.dcir.empty:
                                    dcir = pd.DataFrame({"Cyc": cyccal, "dcir_raw": dcirtemp.dcir})
                                    dcir = dcir.set_index(dcir["Cyc"])
                                df.NewData = pd.concat([Dchg, Ocv, Eff, Chg, DchgEng, Eff2, Temp, AvgV, OriCycle], axis=1).reset_index(drop=True)
                                df.NewData.columns = ["Dchg", "RndV", "Eff", "Chg", "DchgEng", "Eff2", "Temp", "AvgV", "OriCyc"]
                                if isinstance(dcir, pd.Series) and hasattr(dcir, "dcir_raw") and not dcir.dcir_raw.empty:
                                    df.NewData["dcir"] = dcir["dcir_raw"]
                                elif isinstance(dcir, pd.Series):
                                    df.NewData["dcir"] = dcir
                                else:
                                    df.NewData.loc[0, "dcir"] = 0
                            else:
                                df.NewData = pd.concat([Dchg, Ocv, Eff, Chg, DchgEng, Eff2, Temp, AvgV, OriCycle], axis=1).reset_index(drop=True)
                                df.NewData.columns = ["Dchg", "RndV", "Eff", "Chg", "DchgEng", "Eff2", "Temp", "AvgV", "OriCyc"]
                                df.NewData.loc[0, "dcir"] = 0
    return [mincapacity, df]

# PNE Step charge Profile data 처리 class
def pne_step_Profile_data(raw_file_path, inicycle, mincapacity, cutoff, inirate):
    df = pd.DataFrame()
    if (raw_file_path[-4:-1]) != "ter":
        # PNE 채널, 용량 산정
        tempcap = pne_min_cap(raw_file_path, mincapacity, inirate)
        mincapacity = tempcap
        # data 기본 처리
        profile_raw = pne_data(raw_file_path, inicycle)
        # 충전 부분만 별도로 산정
        if hasattr(profile_raw, "Profileraw"):
            profile_raw.Profileraw = profile_raw.Profileraw[(profile_raw.Profileraw[27] == inicycle) 
                                                            & (profile_raw.Profileraw[2].isin([9, 1]))]
            profile_raw.Profileraw = profile_raw.Profileraw[[17, 8, 9, 21, 10, 7]]
            profile_raw.Profileraw.columns = ["PassTime[Sec]", "Voltage[V]", "Current[mA]", "Temp1[Deg]", "Chgcap", "step"]
            # 충전 단위 변환
            profile_raw.Profileraw["PassTime[Sec]"] = profile_raw.Profileraw["PassTime[Sec]"]/100/60
            profile_raw.Profileraw["Voltage[V]"] = profile_raw.Profileraw["Voltage[V]"]/1000000
            if ('PNE21' in raw_file_path) or ('PNE22' in raw_file_path):
                profile_raw.Profileraw["Current[mA]"] = profile_raw.Profileraw["Current[mA]"]/mincapacity/1000000
                profile_raw.Profileraw["Chgcap"] = profile_raw.Profileraw["Chgcap"]/mincapacity/1000000
            else:
                profile_raw.Profileraw["Current[mA]"] = profile_raw.Profileraw["Current[mA]"]/mincapacity/1000
                profile_raw.Profileraw["Chgcap"] = profile_raw.Profileraw["Chgcap"]/mincapacity/1000
            profile_raw.Profileraw["Temp1[Deg]"] = profile_raw.Profileraw["Temp1[Deg]"]/1000
            stepmin = profile_raw.Profileraw.step.min()
            stepmax = profile_raw.Profileraw.step.max()
            stepdiv = stepmax - stepmin
            if not np.isnan(stepdiv):
                if stepdiv == 0:
                    df.stepchg = profile_raw.Profileraw
                else:
                    Profiles = [profile_raw.Profileraw.loc[profile_raw.Profileraw.step == stepmin]]
                    for i in range(1, stepdiv + 1):
                        Profiles.append(profile_raw.Profileraw.loc[profile_raw.Profileraw.step == stepmin + i])
                        Profiles[-1]["PassTime[Sec]"] += Profiles[-2]["PassTime[Sec]"].max()
                        Profiles[-1]["Chgcap"] += Profiles[-2]["Chgcap"].max()
                    df.stepchg = pd.concat(Profiles)
        # 전류 cut-off 설정
        if hasattr(df, "stepchg"):
            df.stepchg = df.stepchg[(df.stepchg["Current[mA]"] >= cutoff)]
            df.stepchg = df.stepchg[["PassTime[Sec]", "Chgcap", "Voltage[V]", "Current[mA]",
                                                "Temp1[Deg]"]]
            df.stepchg.columns = ["TimeMin", "SOC", "Vol", "Crate", "Temp"]
    return [mincapacity, df]

# PNE 율별 충전 Profile 처리
def pne_rate_Profile_data(raw_file_path, inicycle, mincapacity, cutoff, inirate):
    df = pd.DataFrame()
    if (raw_file_path[-4:-1]) != "ter":
        # PNE 채널, 용량 산정
        tempcap = pne_min_cap(raw_file_path, mincapacity, inirate)
        mincapacity = tempcap
        # data 기본 처리
        pnetempdata = pne_data(raw_file_path, inicycle)
        if hasattr(pnetempdata, 'Profileraw'):
            Profileraw = pnetempdata.Profileraw
            Profileraw = Profileraw.loc[(Profileraw[27] == inicycle) & (Profileraw[2].isin([9, 1]))]
            Profileraw = Profileraw[[17, 8, 9, 21, 10, 7]]
            Profileraw.columns = ["PassTime[Sec]", "Voltage[V]", "Current[mA]", "Temp1[Deg]", "Chgcap", "step"]
            # 충전 단위 변환
            Profileraw["PassTime[Sec]"] = Profileraw["PassTime[Sec]"]/100/60
            Profileraw["Voltage[V]"] = Profileraw["Voltage[V]"]/1000000
            if ('PNE21' in raw_file_path) or ('PNE22' in raw_file_path):
                Profileraw["Current[mA]"] = Profileraw["Current[mA]"]/mincapacity/1000000
                Profileraw["Chgcap"] = Profileraw["Chgcap"]/mincapacity/1000000
            else:
                Profileraw["Current[mA]"] = Profileraw["Current[mA]"]/mincapacity/1000
                Profileraw["Chgcap"] = Profileraw["Chgcap"]/mincapacity/1000
            Profileraw["Temp1[Deg]"] = Profileraw["Temp1[Deg]"]/1000
            df.rateProfile = Profileraw
            # cut-off
            if hasattr(df, "rateProfile"):
                df.rateProfile = df.rateProfile[(df.rateProfile["Current[mA]"] >= cutoff)]
                df.rateProfile = df.rateProfile[["PassTime[Sec]", "Chgcap", "Voltage[V]", "Current[mA]", "Temp1[Deg]"]]
                df.rateProfile.columns = ["TimeMin", "SOC", "Vol", "Crate", "Temp"]
    return [mincapacity, df]

# PNE 충전 Profile 처리
def pne_chg_Profile_data(raw_file_path, inicycle, mincapacity, cutoff, inirate, smoothdegree):
    df = pd.DataFrame()
    if (raw_file_path[-4:-1]) != "ter":
        # PNE 채널, 용량 산정
        tempcap = pne_min_cap(raw_file_path, mincapacity, inirate)
        mincapacity = tempcap
        # data 기본 처리
        df = pne_data(raw_file_path, inicycle)
        if hasattr(df, 'Profileraw'):
            df.Profileraw = df.Profileraw.loc[(df.Profileraw[27] == inicycle) & (df.Profileraw[2].isin([9, 1]))]
            df.Profileraw = df.Profileraw[[17, 8, 9, 10, 14, 21, 7]]
            df.Profileraw.columns = ["PassTime[Sec]", "Voltage[V]", "Current[mA]", "Chgcap", "Chgwh", "Temp1[Deg]", "step"]
            # 충전 단위 변환
            df.Profileraw["PassTime[Sec]"] = df.Profileraw["PassTime[Sec]"]/100/60
            df.Profileraw["Voltage[V]"] = df.Profileraw["Voltage[V]"]/1000000
            if ('PNE21' in raw_file_path) or ('PNE22' in raw_file_path):
                df.Profileraw["Current[mA]"] = df.Profileraw["Current[mA]"]/mincapacity/1000000
                df.Profileraw["Chgcap"] = df.Profileraw["Chgcap"]/mincapacity/1000000
            else:
                df.Profileraw["Current[mA]"] = df.Profileraw["Current[mA]"]/mincapacity/1000
                df.Profileraw["Chgcap"] = df.Profileraw["Chgcap"]/mincapacity/1000
            df.Profileraw["Temp1[Deg]"] = df.Profileraw["Temp1[Deg]"]/1000
            stepmin = df.Profileraw.step.min()
            stepmax = df.Profileraw.step.max()
            stepdiv = stepmax - stepmin
            if not np.isnan(stepdiv):
                if stepdiv == 0:
                    df.Profile = df.Profileraw
                else:
                    Profiles = [df.Profileraw.loc[df.Profileraw.step == stepmin]]
                    for i in range(1, stepdiv + 1):
                        Profiles.append(df.Profileraw.loc[df.Profileraw.step == stepmin + i])
                        Profiles[-1]["PassTime[Sec]"] += Profiles[-2]["PassTime[Sec]"].max()
                        Profiles[-1]["Chgcap"] += Profiles[-2]["Chgcap"].max()
                    df.Profile = pd.concat(Profiles)
        if hasattr(df, "Profile"):
            df.Profile = df.Profile.reset_index()
            # cut-off
            df.Profile = df.Profile[(df.Profile["Current[mA]"] >= cutoff)]
            # 충전 용량 산정, dQdV 산정
            df.Profile["dVdQ"] = 0
            df.Profile["delcap"] = 0
            df.Profile["delvol"] = 0
            # 충전 용량 산정, dQdV 산정
            if smoothdegree == 0:
                smoothdegree = len(df.Profile) / 30
            df.Profile["delvol"] = df.Profile["Voltage[V]"].diff(periods=smoothdegree)
            df.Profile["delcap"] = df.Profile["Chgcap"].diff(periods=smoothdegree)
            df.Profile["dQdV"] = df.Profile["delcap"]/df.Profile["delvol"]
            df.Profile["dVdQ"] = df.Profile["delvol"]/df.Profile["delcap"]
            df.Profile = df.Profile[["PassTime[Sec]", "Chgcap", "Chgwh", "Voltage[V]", "Current[mA]",
                                                "dQdV", "dVdQ", "Temp1[Deg]"]]
            df.Profile.columns = ["TimeMin", "SOC", "Energy", "Vol", "Crate", "dQdV", "dVdQ", "Temp"]
    return [mincapacity, df]

# PNE 방전 Profile 처리
def pne_dchg_Profile_data(raw_file_path, inicycle, mincapacity, cutoff, inirate, smoothdegree):
    df = pd.DataFrame()
    if (raw_file_path[-4:-1]) != "ter":
        # PNE 채널, 용량 산정
        tempcap = pne_min_cap(raw_file_path, mincapacity, inirate)
        mincapacity = tempcap
        # data 기본 처리
        pnetempdata = pne_data(raw_file_path, inicycle)
        if hasattr(pnetempdata, 'Profileraw'):
            Profileraw = pnetempdata.Profileraw
            Profileraw = Profileraw.loc[(Profileraw[27] == inicycle) & (Profileraw[2].isin([9, 2]))]
            Profileraw = Profileraw[[17, 8, 9, 11, 15, 21, 7]]
            Profileraw.columns = ["PassTime[Sec]", "Voltage[V]", "Current[mA]", "Dchgcap", "Dchgwh", "Temp1[Deg]", "step"]
            # 충전 단위 변환
            Profileraw["PassTime[Sec]"] = Profileraw["PassTime[Sec]"]/100/60
            Profileraw["Voltage[V]"] = Profileraw["Voltage[V]"]/1000000
            if ('PNE21' in raw_file_path) or ('PNE22' in raw_file_path):
                Profileraw["Current[mA]"] = Profileraw["Current[mA]"]/mincapacity/1000000 * (-1)
                Profileraw["Dchgcap"] = Profileraw["Dchgcap"]/mincapacity/1000000
            else:
                Profileraw["Current[mA]"] = Profileraw["Current[mA]"]/mincapacity/1000 * (-1)
                Profileraw["Dchgcap"] = Profileraw["Dchgcap"]/mincapacity/1000
            Profileraw["Temp1[Deg]"] = Profileraw["Temp1[Deg]"]/1000
            stepmin = Profileraw.step.min()
            stepmax = Profileraw.step.max()
            stepdiv = stepmax - stepmin
            if not np.isnan(stepdiv):
                if stepdiv == 0:
                    df.Profile = Profileraw
                else:
                    Profiles = [Profileraw.loc[Profileraw.step == stepmin]]
                    for i in range(1, stepdiv + 1):
                        Profiles.append(Profileraw.loc[Profileraw.step == stepmin + i])
                        Profiles[-1]["PassTime[Sec]"] += Profiles[-2]["PassTime[Sec]"].max()
                        Profiles[-1]["Dchgcap"] += Profiles[-2]["Dchgcap"].max()
                    df.Profile = pd.concat(Profiles)
        if hasattr(df, 'Profile'):
            df.Profile = df.Profile.reset_index()
            # cut-off
            df.Profile = df.Profile[(df.Profile["Voltage[V]"] >= cutoff)]
            # 충전 용량 산정, dQdV 산정
            df.Profile["dQdV"] = 0
            df.Profile["dVdQ"] = 0
            df.Profile["delcap"] = 0
            df.Profile["delvol"] = 0
            # 충전 용량 산정, dQdV 산정
            if smoothdegree == 0:
                smoothdegree = len(df.Profile) / 30
            df.Profile["delvol"] = df.Profile["Voltage[V]"].diff(periods=smoothdegree)
            df.Profile["delcap"] = df.Profile["Dchgcap"].diff(periods=smoothdegree)
            df.Profile["dQdV"] = df.Profile["delcap"]/df.Profile["delvol"]
            df.Profile["dVdQ"] = df.Profile["delvol"]/df.Profile["delcap"]
            df.Profile = df.Profile[["PassTime[Sec]", "Dchgcap", "Dchgwh", "Voltage[V]", "Current[mA]",
                                                "dQdV", "dVdQ", "Temp1[Deg]"]]
            df.Profile.columns = ["TimeMin", "SOC", "Energy", "Vol", "Crate", "dQdV", "dVdQ", "Temp"]
    return [mincapacity, df]

# PNE continous data scale 변경
def pne_continue_profile_scale_change(raw_file_path, df, mincapacity):
    #단위 변환
    df = df.reset_index()
    df["TotTime[Day]"] = df["TotTime[Day]"] * 8640000
    df["TotTime[Sec]"] = (df["TotTime[Sec]"] + df["TotTime[Day]"]) / 100
    # 시작값 0으로 변경
    df["TotTime[Sec]"] = (df["TotTime[Sec]"] - df.loc[0, "TotTime[Sec]"])
    df["TotTime[Min]"] = (df["TotTime[Sec]"]/60)
    df["Voltage[V]"] = df["Voltage[V]"]/1000000
    if ('PNE21' in raw_file_path) or ('PNE22' in raw_file_path):
        df["Crate"] = (df["Current[mA]"]/mincapacity/1000000).round(2)
        df["Current[mA]"] = (df["Current[mA]"]/1000000000)
        df["ChgCap"] = df["ChgCap"]/mincapacity/1000000
        df["DchgCap"] = df["DchgCap"]/mincapacity/1000000
    else:
        df["Crate"] = (df["Current[mA]"]/mincapacity/1000).round(2)
        df["Current[mA]"] = (df["Current[mA]"]/1000000)
        df["ChgCap"] = df["ChgCap"]/mincapacity/1000
        df["DchgCap"] = df["DchgCap"]/mincapacity/1000
    df["SOC"] = df["DchgCap"] + df["ChgCap"]
    df["Temp1[Deg]"] = df["Temp1[Deg]"]/1000
    df["StepTime"] = df["StepTime"]/100
    return df

# PNE 연속 data 처리 class
def pne_Profile_continue_data(raw_file_path, inicycle, endcycle, mincapacity, inirate, CDstate):
    '''0:Index 1:Stepmode(1:CC-CV, 2:CC, 3:CV, 4:OCV) 2:StepType(0, 1:충전,2:방전,3:휴지,4: OCV, 5: Impedance, 6: End, 8:loop)
    3:ChgDchg 4:State 5:Loop 255:Pattern (Loop:1)
    6:Code(66:충전,65:방전,64:휴지,64:loop) 7:StepNo 8:Voltage(uV) 9:Current(uA) 10:Chg Capacity(uAh)
    11:Dchg Capacity(uAh) 12:Chg Power(uW) 13:Dchg Power(uW) 14:Chg WattHour(Wh) 15:Dchg WattHour(Wh) 
    16: 17:StepTime(/100s) 18:TotTime(day) 19:TotTime(/100s) 20:imp 
    21:Temp1 22:Temp2 23:Temp3 24:Temperature(°C) 25: 26: 27:Total Cycle 28:CurrCycle 29:Average voltage(mV) 30:Average current(A) 
    31: 32: 33:date 34:time 35: 36: 37: 38: 39: 40: 
    41: 42: 43: 44:누적step(Loop, 완료 제외) 45:voltage max 46: '''
    df = pd.DataFrame()
    if (raw_file_path[-4:-1]) != "ter":
        if CDstate != "":
            # PNE 채널, 용량 산정
            tempcap = pne_min_cap(raw_file_path, mincapacity, inirate)
            mincapacity = tempcap
            # data 기본 처리
            pneProfile = pne_continue_data(raw_file_path, inicycle, endcycle)
            if hasattr(pneProfile, 'Profileraw'):
                # Profile 데이터를 기준으로 산정 
                Profileraw = pneProfile.Profileraw
                if CDstate == "CHG":
                    Profileraw = Profileraw.loc[(Profileraw[27] >= inicycle) & (Profileraw[27] <= endcycle) & Profileraw[2].isin([9,1])]
                elif (CDstate == "DCHG") or (CDstate == "DCH"):
                    Profileraw = Profileraw.loc[(Profileraw[27] >= inicycle) & (Profileraw[27] <= endcycle) & Profileraw[2].isin([9,2])]
                elif (CDstate == "Cycle") or (CDstate == "7cyc") or (CDstate == "GITT"):
                    Profileraw = Profileraw.loc[(Profileraw[27] >= inicycle) & (Profileraw[27] <= endcycle)]
                Profileraw = Profileraw[[0, 18, 19, 8, 9, 21, 10, 11, 7, 17]]
                Profileraw.columns = ["index", "TotTime[Day]", "TotTime[Sec]", "Voltage[V]", "Current[mA]", "Temp1[Deg]",
                                            "ChgCap", "DchgCap", "step", "StepTime"]
                Profileraw = pne_continue_profile_scale_change(raw_file_path, Profileraw, mincapacity)
                df.stepchg = Profileraw
                if hasattr(df, "stepchg"):
                    df.stepchg = df.stepchg[["TotTime[Sec]", "TotTime[Min]", "SOC", "Voltage[V]", "Current[mA]", "Crate", "Temp1[Deg]"]]
                    df.stepchg.columns = ["TimeSec", "TimeMin", "SOC", "Vol","Curr", "Crate", "Temp"]
            CycfileSOC = pd.DataFrame()
        else:
            # PNE 채널, 용량 산정
            tempcap = pne_min_cap(raw_file_path, mincapacity, inirate)
            mincapacity = tempcap
            # data 기본 처리
            pneProfile = pne_continue_data(raw_file_path, inicycle, endcycle)
            pnecyc = pne_cyc_continue_data(raw_file_path)
            if hasattr(pnecyc, "Cycrawtemp") and hasattr(pneProfile, 'Profileraw'):
                # cycle 데이터를 기준으로 OCV, CCV 데이터 확인
                pnecyc.Cycrawtemp = pnecyc.Cycrawtemp.loc[(pnecyc.Cycrawtemp[27] >= inicycle) & (pnecyc.Cycrawtemp[27] <= endcycle)]
                CycfileCap =  pnecyc.Cycrawtemp.loc[((pnecyc.Cycrawtemp[2] == 1) | (pnecyc.Cycrawtemp[2] == 2)), [0, 8, 10, 11]]
                CycfileCap.loc[:,"AccCap"] = (CycfileCap.loc[:,10].cumsum() - CycfileCap[11].cumsum())
                CycfileCap = CycfileCap.reset_index()
                CycfileCap.loc[:,"AccCap"] = (CycfileCap.loc[:,"AccCap"] - CycfileCap.loc[0,"AccCap"])/1000
                CycfileOCV =  pnecyc.Cycrawtemp.loc[(pnecyc.Cycrawtemp[2] == 3), [0, 8]]
                CycfileCCV =  pnecyc.Cycrawtemp.loc[((pnecyc.Cycrawtemp[2] == 1) | (pnecyc.Cycrawtemp[2] == 2)), [0, 8]]
                Cycfileraw = pd.merge(CycfileOCV, CycfileCCV, on = 0, how='outer')
                # Cap, OCV, CCV table 별도 산정
                tempCap = CycfileCap.loc[:,"AccCap"].dropna(axis=0).tolist()
                Cap = [abs(i/mincapacity) for i in tempCap]
                tempOCV = CycfileOCV[8].dropna(axis=0).tolist()
                OCV = [i/1000000 for i in tempOCV]
                tempCCV = CycfileCCV[8].dropna(axis=0).tolist()
                CCV = [i/1000000 for i in tempCCV]
                min_length = min(len(Cap), len(OCV), len(CCV))
                CycfileSOC = pd.DataFrame({"AccCap": Cap[:min_length], "OCV": OCV[:min_length], "CCV": CCV[:min_length]})
                # Profile 데이터를 기준으로 산정 
                Profileraw = pneProfile.Profileraw
                Profileraw = Profileraw.loc[(Profileraw[27] >= inicycle) & (Profileraw[27] <= endcycle)]
                Profileraw = Profileraw[[0, 18, 19, 8, 9, 21, 10, 11, 7, 17]]
                Profileraw = pd.merge(Profileraw, Cycfileraw, on = 0, how = 'outer')
                Profileraw.columns = ["index", "TotTime[Day]", "TotTime[Sec]", "Voltage[V]", "Current[mA]", "Temp1[Deg]",
                                            "ChgCap", "DchgCap", "step", "StepTime", "OCV", "CCV"]
                Profileraw["OCV"] = Profileraw["OCV"]/1000000
                Profileraw["CCV"] = Profileraw["CCV"]/1000000
                Profileraw = pne_continue_profile_scale_change(raw_file_path, Profileraw, mincapacity)
                df.stepchg = Profileraw
                if hasattr(df, "stepchg"):
                    df.stepchg = df.stepchg[["TotTime[Sec]", "TotTime[Min]", "SOC", "Voltage[V]", "Current[mA]", "Crate",
                                                        "Temp1[Deg]", "OCV", "CCV"]]
                    df.stepchg.columns = ["TimeSec", "TimeMin", "SOC", "Vol","Curr", "Crate", "Temp", "OCV", "CCV"]
    return [mincapacity, df, CycfileSOC]

# PNE DCIR data 처리 class
def pne_dcir_chk_cycle(raw_file_path):
    '''0:Index 1:Stepmode(1:CC-CV, 2:CC, 3:CV, 4:OCV) 2:StepType(1:충전,2:방전,3:휴지,4: OCV, 5: Impedance, 6: End, 8:loop)
    3:ChgDchg 4:State 5:Loop(Loop:1)
    6:Code(66:충전,65:방전,64:휴지,64:loop) 7:StepNo 8:Voltage(uV) 9:Current(uA) 10:Chg Capacity(uAh)
    11:Dchg Capacity(uAh) 12:Chg Power(uW) 13:Dchg Power(uW) 14:Chg WattHour(Wh) 15:Dchg WattHour(Wh) 
    16: 17:StepTime(/100s) 18:TotTime(day) 19:TotTime(/100s) 20:imp 
    21:Temp1 22:Temp2 23:Temp3 24:Temperature(°C) 25: 26: 27:Total Cycle 28:CurrCycle 29:Average voltage(mV) 30:Average current(A) 
    31: 32: 33:date 34:time 35: 36: 37: 38: 39: 40: 
    41: 42: 43: 44:누적step(Loop, 완료 제외) 45:voltage max 46: '''
    df = pd.DataFrame()
    if (raw_file_path[-4:-1]) != "ter":
        # PNE 채널, 용량 산정
        # data 기본 처리
        pne_dcir_chk = pne_cyc_continue_data(raw_file_path)
        df = pne_dcir_chk.Cycrawtemp
        df = df[[27, 2, 10, 11, 8, 20, 45, 15, 17, 9, 24, 29, 6]]
        df.columns = ["TotlCycle", "Condition", "chgCap", "DchgCap", "Ocv", "imp", "volmax",
                            "DchgEngD", "steptime", "Curr", "Temp", "AvgV", "EndState"]
        # 조건에 맞는 데이터 필터링
        filtered_df = df[(df['Condition'] == 2) & (df['EndState'] == 64) & (df['steptime'] == 2000)]
        filtered_df2 = df[(df['Condition'] == 1) & (df['EndState'] == 64) & (df['steptime'] == 2000)]

        # TotlCycle의 최소값과 최대값 계산
        min_value = filtered_df['TotlCycle'].min()
        max_value = filtered_df['TotlCycle'].max()
        result = [f"{min_value}-{max_value}"]
        if not filtered_df2.empty:
            min_value2 = filtered_df2['TotlCycle'].min()
            max_value2 = filtered_df2['TotlCycle'].max()
            result.append(f"{min_value2}-{max_value2}")

        # 결과 반환
        return result

# PNE DCIR data 처리 class
def pne_dcir_Profile_data(raw_file_path, inicycle, endcycle, mincapacity, inirate):
    '''0:Index 1:Stepmode(1:CC-CV, 2:CC, 3:CV, 4:OCV) 2:StepType(1:충전,2:방전,3:휴지,4: OCV, 5: Impedance, 6: End, 8:loop)
    3:ChgDchg 4:State 5:Loop(Loop:1)
    6:Code(66:충전,65:방전,64:휴지,64:loop) 7:StepNo 8:Voltage(uV) 9:Current(uA) 10:Chg Capacity(uAh)
    11:Dchg Capacity(uAh) 12:Chg Power(uW) 13:Dchg Power(uW) 14:Chg WattHour(Wh) 15:Dchg WattHour(Wh) 
    16: 17:StepTime(/100s) 18:TotTime(day) 19:TotTime(/100s) 20:imp 
    21:Temp1 22:Temp2 23:Temp3 24:Temperature(°C) 25: 26: 27:Total Cycle 28:CurrCycle 29:Average voltage(mV) 30:Average current(A) 
    31: 32: 33:date 34:time 35: 36: 37: 38: 39: 40: 
    41: 42: 43: 44:누적step(Loop, 완료 제외) 45:voltage max 46: '''
    df = pd.DataFrame()
    if (raw_file_path[-4:-1]) != "ter":
        # PNE 채널, 용량 산정
        tempcap = pne_min_cap(raw_file_path, mincapacity, inirate)
        mincapacity = tempcap
        # data 기본 처리
        pneProfile = pne_continue_data(raw_file_path, inicycle, endcycle)
        pnecycraw = pne_cyc_continue_data(raw_file_path)
        if hasattr(pneProfile, 'Profileraw'):
            Profileraw = pneProfile.Profileraw
            Profileraw = Profileraw.loc[(Profileraw[27] >= (inicycle)) & (Profileraw[27] <= (endcycle))]
            Profileraw = Profileraw[[0, 18, 19, 8, 9, 21, 10, 11, 7, 27, 17]]
            Profileraw.columns = ["index", "TotTime[Day]", "TotTime[Sec]", "Voltage[V]", "Current[mA]", "Temp1[Deg]", "ChgCap",
                                  "DchgCap", "step", "TotCyc", "StepTime"]
            # 20s 종료되는 step을 기준으로 DCIR step, 전류 산정
            dcir_base = Profileraw.loc[Profileraw["StepTime"] == 20]
            dcir_base.reset_index(drop=True, inplace=True)
            dcir_step = list(set(dcir_base["step"].tolist()))
            # 율별 pulse C-rate 확인
            if ('PNE21' in raw_file_path) or ('PNE22' in raw_file_path):
                dcir_crate = [((dcir_base.loc[i, "Current[mA]"] / 1000000)/mincapacity).round(2) for i in range(0,4)]
            else:
                dcir_crate = [((dcir_base.loc[i, "Current[mA]"] / 1000)/mincapacity).round(2) for i in range(0,4)]
            dcir_crate.sort()
            # DCIR 시간을 0.2초로 변경
            dcir_time = [0.0, 0.3, 1.0, 10.0, 20.0]
            # Profile 데이터를 기준으로 산정 
            # DCIR 스텝과 원하는 시간의 숫자를 확인
            Profileraw = pne_continue_profile_scale_change(raw_file_path, Profileraw, mincapacity)
            Profileraw = Profileraw[Profileraw["step"].isin(dcir_step)]
            Profileraw = Profileraw[Profileraw["StepTime"].isin(dcir_time)]
            Profileraw = Profileraw[Profileraw["Crate"].isin(dcir_crate)]
            Profileraw = Profileraw[["TotTime[Sec]", "TotTime[Min]", "Voltage[V]", "Current[mA]", "Crate", "Temp1[Deg]", "step",
                                     "TotCyc", "StepTime"]]
            Profileraw.columns = ["TimeSec", "TimeMin", "Vol", "Curr", "Crate", "Temp", "step", "Cyc", "StepTime"]
        if hasattr(pnecycraw, "Cycrawtemp"):
            # cycle 데이터를 기준으로 OCV, CCV 데이터 확인
            pnecyc = pnecycraw.Cycrawtemp
            pnecyc2 = pnecycraw.Cycrawtemp
            pnecyc = pnecyc.loc[(pnecyc[27] >= (inicycle - 1)) & (pnecyc[27] <= (endcycle - 1))]
            pnecyc2 = pnecyc2.loc[(pnecyc2[27] >= (inicycle)) & (pnecyc2[27] <= (endcycle))]
            if len(pnecyc) != 0 and len(pnecyc2) != 0:
                CycfileCap =  pnecyc.loc[(pnecyc[2] == 8), [0, 27, 10, 11, 8, 9]]
                real_ocv = pnecyc2.loc[
                    (pnecyc2[2] == 3) & 
                    (pnecyc2[17].isin([360000, 720000, 1080000, 2160000])), 
                    [8]
                ]
                real_ocv = real_ocv.reset_index()
                CycfileCap["AccCap"] = (CycfileCap.loc[:,10].cumsum() - CycfileCap[11].cumsum())
                CycfileCap = CycfileCap.reset_index()
                CycfileCap["AccCap"] = abs((CycfileCap.loc[:,"AccCap"] - CycfileCap.loc[0,"AccCap"])/1000)
                if ('PNE21' in raw_file_path) or ('PNE22' in raw_file_path):
                    CycfileCap["AccCap"] = CycfileCap["AccCap"]/1000
                if dcir_crate[-2] < 0:
                    CycfileCap["SOC"] = (1 - CycfileCap["AccCap"]/mincapacity) * 100
                else:
                    CycfileCap["SOC"] = (CycfileCap["AccCap"]/mincapacity) * 100
                CycfileCap["SOC"] = CycfileCap["SOC"] - (CycfileCap["SOC"].max() - 100) 
                CycfileCap["Cyc"] = (CycfileCap[27])
                CycfileCap["rOCV"] = real_ocv[8]/1000000
                CycfileCap["CCV"] = CycfileCap[8]/1000000
                CycfileCap["curr"] = CycfileCap[9]/1000000
                CycfileCap.loc[0, "CCV"] = np.nan
                CycfileCap["RSS"] = abs((CycfileCap["CCV"] - CycfileCap["rOCV"])/ CycfileCap["curr"]) *1000
                CycfileCap = CycfileCap[["Cyc", "AccCap", "SOC", "CCV", "rOCV", "RSS"]]
                CycfileCap["Cyc"] = CycfileCap["Cyc"] + 1
                full_length = len(CycfileCap)
                RSSfileCap = CycfileCap.copy()
                # sloep base DCIR 계산
                for time in dcir_time:
                    temp_dcir_slope = []
                    temp_dcir_rsq = []
                    temp_dcir_est_ocv = []
                    dcir_slope = []
                    dcir_est_ocv = []
                    # Step 시간별로 pvt table 생성
                    time_dcir = Profileraw[Profileraw["StepTime"] == time]
                    profile_pvt = time_dcir.pivot_table(index="Cyc", columns="Crate", values = "Vol")
                    profile_pvt_curr = time_dcir.pivot_table(index="Cyc", columns="Crate", values = "Curr")
                    if time != 0:
                        for row in range(len(profile_pvt)):
                            pvt_row_vol = profile_pvt.iloc[row].tolist()
                            pvt_row_curr = (profile_pvt_curr.iloc[row] / 1000).tolist()
                            # slope를 DCIR로 산정
                            slope = linregress(pvt_row_curr, pvt_row_vol)[0]
                            # 상관계수 확인
                            rsq = (linregress(pvt_row_curr, pvt_row_vol)[2]) ** 2 * 100
                            if slope > 0:
                                # slope를 DCIR로 산정
                                temp_dcir_slope.append(slope)
                                # rsq를 DCIR로 산정
                                temp_dcir_rsq.append(rsq)
                            if time == dcir_time[1]:
                                # 절편을 OCV로 산정
                                ocv = linregress(pvt_row_curr, pvt_row_vol)[1]
                                temp_dcir_est_ocv.append(ocv)
                        if time == dcir_time[1]:
                            dcir_est_ocv = temp_dcir_est_ocv + [np.nan] * (full_length - len(temp_dcir_est_ocv))
                            CycfileCap["OCV"] = dcir_est_ocv[:len(CycfileCap)]
                            RSSfileCap["OCV"] = dcir_est_ocv[:len(RSSfileCap)]
                        dcir_slope = temp_dcir_slope + [np.nan] * (full_length - len(temp_dcir_slope))
                        dcir_rsq = temp_dcir_rsq + [np.nan] * (full_length - len(temp_dcir_rsq))
                        CycfileCap[str(time)] = dcir_slope[:len(CycfileCap)]
                        RSSfileCap[str(time) + "_rsq"] = dcir_rsq[:len(RSSfileCap)]
                    else:
                        pass
                return [mincapacity, CycfileCap, RSSfileCap]

def set_log_cycle(filename, realcyc, recentno, allcycle, manualcycle, manualcycleno):
    '''
    Set log
    0:[TIME] 1: IMEI 2: Binary version 3: Capacity 4: cisd_fullcaprep_max 5: batt_charging_source
    6: charging_type 7: voltage_now 8: voltage_avg 9: current_now 10: current_avg
    11: battery_temp 12: ac_temp 13: temperature 14: battery_cycle 15: battery_charger_status
    16: batt_slate_mode 17: fg_asoc 18: fg_cycle 19: BIG 20: Little
    21: G3D 22: ISP 23: curr_5 24: wc_vrect 25: wc_vout
    26: dchg_temp 27: dchg_temp_adc 28: direct_charging_iin 29: AP CUR_CH0 30: AP CUR_CH1
    31: AP CUR_CH2 32: AP CUR_CH3 33: AP CUR_CH4 34: AP CUR_CH5 35: AP CUR_CH6
    36: AP CUR_CH7 37: AP POW_CH0 38: AP POW_CH1 39: AP POW_CH2 40: AP POW_CH3
    41: AP POW_CH4 42: AP POW_CH5 43: AP POW_CH6 44: AP POW_CH7 45: cisd_data
    46: LRP 47: USB_TEMP
    '''
    df = pd.DataFrame()
    df.Profile = pd.read_csv(filename, sep=",", engine="c", encoding="UTF-8", on_bad_lines='skip') # IMEI log import
    df.Profile = df.Profile.iloc[:,[0, 3, 7, 9, 6, 11, 14, 6]]
    df.Profile.columns = ["Time", "Level", "Voltage(mV)", "Ctype(Etc)-ChargCur", "Charging", "Temperature(BA)",
                          "Battery_Cycle", "PlugType"]
    set = 2
    df.Profile = df.Profile[:-1]
    cycmin = int(df.Profile.Battery_Cycle.min())
    cycmax = int(df.Profile.Battery_Cycle.max())
    recentcycno = int(recentno)
    # df.Profile['Battery_Cycle_origin'] = df.Profile['Battery_Cycle']
    # 전체 사이클과 최근 사이클 기준 설정
    if allcycle == True:
        cyclecountmax = range(cycmin, cycmax + 1)
    elif manualcycle == True:
        manualcyclenochk = list(map(int, (manualcycleno.text().split())))
        if len(manualcyclenochk) > 2:
            manualcyclenochk = [x for x in manualcyclenochk if (x >= cycmin and x <= cycmax)]
            cyclecountmax = manualcyclenochk
        else:
            cycmin = max(cycmin, manualcyclenochk[0])
            cycmax = min(cycmax, manualcyclenochk[1])
            cyclecountmax = range(cycmin, cycmax + 1)
    else:
    # 최근 20 cycle 기준으로 설정
        if (cycmax - cycmin) > recentcycno:
            df.Profile = df.Profile.loc[(df.Profile["Battery_Cycle"] > (cycmax - recentcycno - 1)) &
                                        (df.Profile["Battery_Cycle"] < (cycmax))]
        else:
            df.Profile = df.Profile.loc[(df.Profile["Battery_Cycle"] > (cycmin)) & (df.Profile["Battery_Cycle"] < (cycmax + 1))]
        df.Profile.reset_index(drop=True, inplace=True)
        
    if realcyc == 0:
        if cycmin != cycmax:
            n = cycmin
            df.Profile['Battery_Cycle'] = n
            for m in range(1, len(df.Profile) - 3):
                if (df.Profile.loc[m, 'PlugType'] == "Unplugged" or df.Profile.loc[m, 'PlugType'] == " NONE") and (
                        df.Profile.loc[m + 1, 'PlugType'] == "AC" or df.Profile.loc[m + 1, 'PlugType'] == " PDIC_APDO") and (
                        df.Profile.loc[m + 2, 'PlugType'] == "AC" or df.Profile.loc[m + 2, 'PlugType'] == " PDIC_APDO") and (
                        df.Profile.loc[m + 3, 'PlugType'] == "AC" or df.Profile.loc[m + 3, 'PlugType'] == " PDIC_APDO"):
                    n = n + 1
                    # df.Profile.loc[m + 1, 'Battery_Cycle'] = n
                df.Profile.loc[m + 1, 'Battery_Cycle'] = n
            df.Profile.loc[len(df.Profile)-2, 'Battery_Cycle'] = n
            df.Profile.loc[len(df.Profile)-1, 'Battery_Cycle'] = n
    cycmax = int(df.Profile.Battery_Cycle.max())
    return [cycmin, cycmax, df]

def set_act_ect_battery_status_cycle(filename, realcyc, recentno, allcycle, manualcycle, manualcycleno):
    '''
    Set log
    0:[TIME] 1: IMEI 2: Binary version 3: Capacity 4: cisd_fullcaprep_max 5: batt_charging_source
    6: charging_type 7: voltage_now 8: voltage_avg 9: current_now 10: current_avg
    11: battery_temp 12: ac_temp 13: temperature 14: battery_cycle 15: battery_charger_status
    16: batt_slate_mode 17: fg_asoc 18: fg_cycle 19: BIG 20: Little
    21: G3D 22: ISP 23: curr_5 24: wc_vrect 25: wc_vout
    26: dchg_temp 27: dchg_temp_adc 28: direct_charging_iin 29: AP CUR_CH0 30: AP CUR_CH1
    31: AP CUR_CH2 32: AP CUR_CH3 33: AP CUR_CH4 34: AP CUR_CH5 35: AP CUR_CH6
    36: AP CUR_CH7 37: AP POW_CH0 38: AP POW_CH1 39: AP POW_CH2 40: AP POW_CH3
    41: AP POW_CH4 42: AP POW_CH5 43: AP POW_CH6 44: AP POW_CH7 45: cisd_data
    46: LRP 47: USB_TEMP

    ECT result
    0: Time 1: voltage_now(mV) 2: Vavg(mV) 3: Ctype(Etc)-ChargCur 4: CurrentAvg.  5: Temperature(BA)
    6: Level 7: Charging 8: Battery_Cycle 9: diffTime 10: compVoltage
    11: ectSOC 12: RSOC 13: SOC_RE 14: SOC_EDV 15: SOH
    16: AnodePotential 17: SOH_dR 18: SOH_CA 19: SOH_X

    Battery Status
    0:Time 1:Level 2:Charging 3:Temperature(BA) 4:PlugType 5:Speed
    6:Voltage(mV) 7:Temperature(CHG) 8:Temperature(AP) 9:Temperature(Coil) 10:Ctype(Etc)-VOL
    11:Ctype(Etc)-ChargCur 12:Ctype(Etc)-Wire_Vout 13:Ctype(Etc)-Wire_Vrect 14:Temperature(CHG ADC) 15:Temperature(Coil ADC)
    16:Temperature(BA ADC) 17:SafetyTimer 18:USB_Thermistor 19:SIOP_Level 20:Battery_Cycle
    21:Fg_Cycle 22:Charge_Time_Remaining 23:IIn 24:Temperature(DC) 25:Temperature(DC ADC)
    26:DC Step 27:DC Status 28:Main Voltage 29:Sub Voltage 30:Main Current Now
    31:Sub Current Now 32:Temperature(SUB Batt) 33:Temperature(SUB Batt ADC) 34:Current Avg.  35:ASOC1
    36:Full Cap Nom 37:ASOC2 38:LRP 39:Raw SOC (%) 40:V avg (mV)
    41:WC_Freq.  42:WC_Tx ID 43:Uno Vout 44:WC_Iin/Iout 45:Power
    46:WC_Rx type 47:BSOH 48:Wireless 2.0 auth status 49:Full Voltage 50:Recharging Voltage
    51:Full Cap Rep 52:CMD DATA 53:Temperature(AP ADC) 54:Battery Cycle Sub 55:charge status
    56:Charging Cable 57:Fan Step 58:Fan Rpm 59:Main Vchg 60:Sub Vchg
    61:err_wthm
    '''
    df = pd.DataFrame()
    if "txt" in filename:
        # ECT 모델 log 관련
        if "Chem" in filename:
            df.Profile = pd.read_csv(filename, sep=",", engine="c", encoding="UTF-8", usecols=[0, 1, 3, 5, 6, 7, 8],
                                     on_bad_lines='skip')
            df.Profile.columns = ['Time', 'Voltage(mV)', 'Ctype(Etc)-ChargCur', 'Temperature(BA)', 'Level', 'Charging',
                                  'Battery_Cycle']
            df.Profile.Time = '20'+ df.Profile['Time'].astype(str)
            df.Profile["Charging"] = df.Profile["Charging"].str.replace(" ","")
            df.Profile["PlugType"] = df.Profile["Charging"]
            df.Profile["PlugType"] = df.Profile["PlugType"].str.replace("Charging", "AC")
            df.Profile["PlugType"] = df.Profile["PlugType"].str.replace("Full", "AC")
            df.Profile["PlugType"] = df.Profile["PlugType"].str.replace("Discharging", "Unplugged")
        # batteryStatus log file 관련
        else:
            # df.Profile = pd.read_csv(filename, sep="\t", engine="c", encoding="UTF-8", skiprows=1)
            df.Profile = pd.read_csv(filename, sep=",", engine="c", encoding="UTF-8", on_bad_lines='skip') # IMEI log import 
    # BSOH log 관련
    elif "xlsx" in filename:
        wb = xw.Book(filename)
        df.Profile = wb.sheets(1).used_range.options(pd.DataFrame, index=False).value
        wb.close()
    else:
        df.Profile = pd.read_csv(filename, sep=",", engine="c", encoding="cp949", on_bad_lines='skip')
        
    if " IMEI" in df.Profile.columns:
        df.Profile = df.Profile.iloc[:,[0, 3, 7, 9, 6, 11, 14, 6]]
        df.Profile.columns = ["Time", "Level", "Voltage(mV)", "Ctype(Etc)-ChargCur", "Charging", "Temperature(BA)",
                              "Battery_Cycle", "PlugType"]
        df.set = 2
    else:
        df.Profile = df.Profile[["Time", "Level", "Voltage(mV)", "Ctype(Etc)-ChargCur", "Charging", "Temperature(BA)",
                                 "Battery_Cycle", "PlugType"]]
        if len(df.Profile.Time[1]) > 18:
            df.set = 3
        elif len(df.Profile.Time[1]) > 10:
            df.set = 1
        else:
            df.set = 0
    df.Profile = df.Profile[:-1]
    cycmin = int(df.Profile.Battery_Cycle.min())
    cycmax = int(df.Profile.Battery_Cycle.max())
    # df.Profile['Battery_Cycle_origin'] = df.Profile['Battery_Cycle']
    recentcycno = int(recentno)
    # 전체 사이클과 최근 사이클 기준 설정
    if allcycle == True:
        cyclecountmax = range(cycmin, cycmax + 1)
    elif manualcycle == True:
        manualcyclenochk = list(map(int, (manualcycleno.text().split())))
        if len(manualcyclenochk) > 2:
            manualcyclenochk = [x for x in manualcyclenochk if (x >= cycmin and x <= cycmax)]
            cyclecountmax = manualcyclenochk
        else:
            cycmin = max(cycmin, manualcyclenochk[0])
            cycmax = min(cycmax, manualcyclenochk[1])
            cyclecountmax = range(cycmin, cycmax + 1)
    else:
    # 최근 20 cycle 기준으로 설정
        if (cycmax - cycmin) > recentcycno:
            df.Profile = df.Profile.loc[(df.Profile["Battery_Cycle"] > (cycmax - recentcycno)) &
                                        (df.Profile["Battery_Cycle"] < (cycmax + 1))]
        elif cycmax == cycmin:
            pass
        else:
            df.Profile = df.Profile.loc[(df.Profile["Battery_Cycle"] > (cycmin)) & (df.Profile["Battery_Cycle"] < (cycmax + 1))]
        df.Profile.reset_index(drop=True, inplace=True)
    if realcyc == 0:
        if cycmin != cycmax:
            n = cycmin
            df.Profile['Battery_Cycle'] = n
            for m in range(1, len(df.Profile) - 3):
                if (df.Profile.loc[m, 'PlugType'] == "Unplugged" or df.Profile.loc[m, 'PlugType'] == " NONE") and (
                        df.Profile.loc[m + 1, 'PlugType'] == "AC" or df.Profile.loc[m + 1, 'PlugType'] == " PDIC_APDO") and (
                        df.Profile.loc[m + 2, 'PlugType'] == "AC" or df.Profile.loc[m + 2, 'PlugType'] == " PDIC_APDO") and (
                        df.Profile.loc[m + 3, 'PlugType'] == "AC" or df.Profile.loc[m + 3, 'PlugType'] == " PDIC_APDO"):
                    n = n + 1
                    # df.Profile.loc[m + 1, 'Battery_Cycle'] = n
                df.Profile.loc[m + 1, 'Battery_Cycle'] = n
            df.Profile.loc[len(df.Profile)-2, 'Battery_Cycle'] = n
            df.Profile.loc[len(df.Profile)-1, 'Battery_Cycle'] = n
    cycmax = int(df.Profile.Battery_Cycle.max())
    return [cycmin, cycmax, df]

def set_act_log_Profile(rawdatafile, mincapacity, selectcyc):
    df = pd.DataFrame()
    df.Profile = rawdatafile
    df.Profile.columns = ["Time", "SOC", "Vol", "Curr", "Type", "Temp", "Cyc", "State"]
    df.Profile = df.Profile[(df.Profile.Cyc == selectcyc)]
    if not df.Profile.empty:
        df.Profile = df.Profile.reset_index()
        # 시간 확인
        if len(df.Profile.Time[1]) > 18:
            df.Profile.Time = pd.to_datetime(df.Profile.Time, format="%Y-%m-%d %H:%M")
        elif len(df.Profile.Time[1]) > 10:
            df.Profile.Time = pd.to_datetime(df.Profile.Time, format="%m/%d %H:%M")
        # 시간 변환
        df.Profile.Time = df.Profile.Time - df.Profile.Time.loc[0]
        df.Profile.Time = df.Profile.Time.dt.total_seconds().div(3600).astype(float)
        df.Profile.Vol = df.Profile.Vol/1000000
        df.Profile.Curr = df.Profile.Curr/mincapacity/1000
        df.Profile.Temp = df.Profile.Temp/10
        df.Profile["delTime"] = 0
        df.Profile["delCap"] = 0
        df.Profile["SOC2"] = 0
        df.DchgProfile = df.Profile[df.Profile.Type == " NONE"]
        df.ChgProfile = df.Profile[df.Profile.Type != " NONE"]
        if not df.DchgProfile.empty:
            df.DchgProfile = df.DchgProfile.reset_index()
            df.DchgProfile.Time = df.DchgProfile.Time - df.DchgProfile.Time[0]
            df.DchgProfile.delTime = df.DchgProfile.Time.diff()
            df.DchgProfile.delCap = df.DchgProfile.delTime * df.DchgProfile.Curr
            df.DchgProfile.SOC2 = df.DchgProfile.delCap.cumsum() * -100
        if not df.ChgProfile.empty:
            df.ChgProfile = df.ChgProfile.reset_index()
            df.ChgProfile.Time = df.ChgProfile.Time - df.ChgProfile.Time[0]
            df.ChgProfile.delTime = df.ChgProfile.Time.diff()
            df.ChgProfile.delCap = df.ChgProfile.delTime * df.ChgProfile.Curr
            df.ChgProfile.SOC2 = df.ChgProfile.delCap.cumsum() * 100
    return df

def set_battery_status_log_Profile(rawdatafile, mincapacity, selectcyc, setcond):
    df = pd.DataFrame()
    df.Profile = rawdatafile
    df.Profile.columns = ["Time", "SOC", "Vol", "Curr", "Type", "Temp", "Cyc", "State"]
    df.Profile = df.Profile[(df.Profile.Cyc == selectcyc)]
    if not df.Profile.empty:
        # 시간 확인
        if setcond == 1:
            df.Profile.Time = pd.to_datetime(df.Profile.Time, format="%Y%m%d %H:%M:%S")
        elif setcond == 0:
            df.Profile.Time = pd.to_datetime(df.Profile.Time, format="%H:%M:%S")
        elif setcond == 3:
            df.Profile.Time = pd.to_datetime(df.Profile.Time, format="%Y%m%d %H:%M:%S.%f")
        elif setcond == 4:
            df.Profile.Time = pd.to_datetime(df.Profile.Time, format="%m%d %H:%M")
        # 시간 변환
        if setcond ==2:
            df.Profile = df.Profile.reset_index()
            df.Profile.Time = df.Profile.Time - df.Profile.Time.loc[0]
            df.Profile.Time = df.Profile.Time.dt.total_seconds().div(3600).astype(float)
            df.Profile.Vol = df.Profile.Vol/1000000
            df.Profile.Curr = df.Profile.Curr/mincapacity/1000
            df.Profile.Temp = df.Profile.Temp/10
            df.Profile["delTime"] = 0
            df.Profile["delCap"] = 0
            df.Profile["SOC2"] = 0
            df.DchgProfile = df.Profile[df.Profile.Type == " NONE"]
            df.ChgProfile = df.Profile[df.Profile.Type != " NONE"]
        elif setcond ==3:
            df.Profile = df.Profile.reset_index()
            df.Profile.Time = df.Profile.Time - df.Profile.Time.loc[0]
            df.Profile.Time = df.Profile.Time.dt.total_seconds().div(3600).astype(float)
            df.Profile.Vol = df.Profile.Vol/1000
            df.Profile.Curr = df.Profile.Curr/mincapacity
            df.Profile["delTime"] = 0
            df.Profile["delCap"] = 0
            df.Profile["SOC2"] = 0
            df.DchgProfile = df.Profile[df.Profile.Type == "Discharging"]
            df.ChgProfile = df.Profile[df.Profile.Type != "Discharging"]
        else:
            df.Profile = df.Profile.reset_index()
            df.Profile.Time = df.Profile.Time - df.Profile.Time.loc[0]
            df.Profile.Time = df.Profile.Time.dt.total_seconds().div(3600).astype(float)
            df.Profile.Vol = df.Profile.Vol/1000
            df.Profile.Curr = df.Profile.Curr/mincapacity
            df.Profile["delTime"] = 0
            df.Profile["delCap"] = 0
            df.Profile["SOC2"] = 0
            df.DchgProfile = df.Profile[df.Profile.Type == "Discharging"]
            df.ChgProfile = df.Profile[df.Profile.Type != "Discharging"]
        if not df.DchgProfile.empty:
            df.DchgProfile = df.DchgProfile.reset_index()
            df.DchgProfile.Time = df.DchgProfile.Time - df.DchgProfile.Time[0]
            df.DchgProfile.delTime = df.DchgProfile.Time.diff()
            df.DchgProfile.delCap = df.DchgProfile.delTime * df.DchgProfile.Curr
            df.DchgProfile.SOC2 = df.DchgProfile.delCap.cumsum() * -100
        if not df.ChgProfile.empty:
            df.ChgProfile = df.ChgProfile.reset_index()
            df.ChgProfile.Time = df.ChgProfile.Time - df.ChgProfile.Time[0]
            df.ChgProfile.delTime = df.ChgProfile.Time.diff()
            df.ChgProfile.delCap = df.ChgProfile.delTime * df.ChgProfile.Curr
            df.ChgProfile.SOC2 = df.ChgProfile.delCap.cumsum() * 100
    return df

class Ui_sitool(object):
    def setupUi(self, sitool):
        sitool.setObjectName("sitool")
        sitool.resize(1913, 1005)
        font = QtGui.QFont()
        font.setFamily("malgun gothic")
        font.setPointSize(10)
        sitool.setFont(font)
        self.layoutWidget = QtWidgets.QWidget(parent=sitool)
        self.layoutWidget.setGeometry(QtCore.QRect(12, 12, 1894, 984))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.layoutWidget.setFont(font)
        self.layoutWidget.setObjectName("layoutWidget")
        self.verticalLayout_39 = QtWidgets.QVBoxLayout(self.layoutWidget)
        self.verticalLayout_39.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_39.setObjectName("verticalLayout_39")
        self.verticalLayout_38 = QtWidgets.QVBoxLayout()
        self.verticalLayout_38.setObjectName("verticalLayout_38")
        self.tabWidget = QtWidgets.QTabWidget(parent=self.layoutWidget)
        self.tabWidget.setMinimumSize(QtCore.QSize(1890, 885))
        self.tabWidget.setMaximumSize(QtCore.QSize(1890, 885))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.tabWidget.setFont(font)
        self.tabWidget.setTabPosition(QtWidgets.QTabWidget.TabPosition.North)
        self.tabWidget.setTabShape(QtWidgets.QTabWidget.TabShape.Rounded)
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.horizontalLayout_123 = QtWidgets.QHBoxLayout(self.tab)
        self.horizontalLayout_123.setObjectName("horizontalLayout_123")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.horizontalLayout_11 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_11.setObjectName("horizontalLayout_11")
        self.tb_room = QtWidgets.QComboBox(parent=self.tab)
        self.tb_room.setMinimumSize(QtCore.QSize(250, 40))
        self.tb_room.setMaximumSize(QtCore.QSize(250, 40))
        self.tb_room.setBaseSize(QtCore.QSize(10, 0))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.tb_room.setFont(font)
        self.tb_room.setObjectName("tb_room")
        self.tb_room.addItem("")
        self.tb_room.addItem("")
        self.tb_room.addItem("")
        self.tb_room.addItem("")
        self.tb_room.addItem("")
        self.horizontalLayout_11.addWidget(self.tb_room)
        self.tb_cycler = QtWidgets.QComboBox(parent=self.tab)
        self.tb_cycler.setMinimumSize(QtCore.QSize(250, 40))
        self.tb_cycler.setMaximumSize(QtCore.QSize(250, 40))
        self.tb_cycler.setBaseSize(QtCore.QSize(10, 0))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.tb_cycler.setFont(font)
        self.tb_cycler.setMaxVisibleItems(35)
        self.tb_cycler.setObjectName("tb_cycler")
        self.tb_cycler.addItem("")
        self.tb_cycler.addItem("")
        self.tb_cycler.addItem("")
        self.tb_cycler.addItem("")
        self.tb_cycler.addItem("")
        self.tb_cycler.addItem("")
        self.tb_cycler.addItem("")
        self.tb_cycler.addItem("")
        self.tb_cycler.addItem("")
        self.tb_cycler.addItem("")
        self.horizontalLayout_11.addWidget(self.tb_cycler)
        self.tb_info = QtWidgets.QComboBox(parent=self.tab)
        self.tb_info.setMinimumSize(QtCore.QSize(450, 40))
        self.tb_info.setMaximumSize(QtCore.QSize(450, 40))
        self.tb_info.setBaseSize(QtCore.QSize(10, 0))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.tb_info.setFont(font)
        self.tb_info.setObjectName("tb_info")
        self.tb_info.addItem("")
        self.tb_info.addItem("")
        self.tb_info.addItem("")
        self.tb_info.addItem("")
        self.tb_info.addItem("")
        self.tb_info.addItem("")
        self.tb_info.addItem("")
        self.tb_info.addItem("")
        self.tb_info.addItem("")
        self.tb_info.addItem("")
        self.horizontalLayout_11.addWidget(self.tb_info)
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        spacerItem = QtWidgets.QSpacerItem(342, 40, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.horizontalLayout_10.addItem(spacerItem)
        self.label_9 = QtWidgets.QLabel(parent=self.tab)
        self.label_9.setMinimumSize(QtCore.QSize(58, 40))
        self.label_9.setMaximumSize(QtCore.QSize(58, 40))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.label_9.setFont(font)
        self.label_9.setObjectName("label_9")
        self.horizontalLayout_10.addWidget(self.label_9)
        self.FindText = QtWidgets.QLineEdit(parent=self.tab)
        self.FindText.setMinimumSize(QtCore.QSize(480, 40))
        self.FindText.setMaximumSize(QtCore.QSize(480, 40))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.FindText.setFont(font)
        self.FindText.setInputMask("")
        self.FindText.setObjectName("FindText")
        self.horizontalLayout_10.addWidget(self.FindText)
        self.horizontalLayout_11.addLayout(self.horizontalLayout_10)
        self.verticalLayout_5.addLayout(self.horizontalLayout_11)
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.tb_summary = QtWidgets.QTableWidget(parent=self.tab)
        self.tb_summary.setMinimumSize(QtCore.QSize(220, 70))
        self.tb_summary.setMaximumSize(QtCore.QSize(220, 70))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.tb_summary.setFont(font)
        self.tb_summary.setLayoutDirection(QtCore.Qt.LayoutDirection.LeftToRight)
        self.tb_summary.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.tb_summary.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.tb_summary.setAutoScroll(False)
        self.tb_summary.setDefaultDropAction(QtCore.Qt.DropAction.IgnoreAction)
        self.tb_summary.setTextElideMode(QtCore.Qt.TextElideMode.ElideRight)
        self.tb_summary.setRowCount(2)
        self.tb_summary.setColumnCount(1)
        self.tb_summary.setObjectName("tb_summary")
        item = QtWidgets.QTableWidgetItem()
        self.tb_summary.setVerticalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tb_summary.setVerticalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.tb_summary.setItem(0, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.tb_summary.setItem(1, 0, item)
        self.tb_summary.horizontalHeader().setVisible(False)
        self.tb_summary.horizontalHeader().setDefaultSectionSize(160)
        self.tb_summary.horizontalHeader().setMinimumSectionSize(160)
        self.tb_summary.verticalHeader().setDefaultSectionSize(33)
        self.tb_summary.verticalHeader().setMinimumSectionSize(33)
        self.horizontalLayout_8.addWidget(self.tb_summary)
        self.tableWidget = QtWidgets.QTableWidget(parent=self.tab)
        self.tableWidget.setMinimumSize(QtCore.QSize(1630, 80))
        self.tableWidget.setMaximumSize(QtCore.QSize(1630, 80))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.tableWidget.setFont(font)
        self.tableWidget.setRowCount(3)
        self.tableWidget.setColumnCount(6)
        self.tableWidget.setObjectName("tableWidget")
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setItem(0, 0, item)
        item = QtWidgets.QTableWidgetItem()
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.BrushStyle.NoBrush)
        item.setForeground(brush)
        self.tableWidget.setItem(0, 1, item)
        item = QtWidgets.QTableWidgetItem()
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.BrushStyle.NoBrush)
        item.setBackground(brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 255))
        brush.setStyle(QtCore.Qt.BrushStyle.SolidPattern)
        item.setForeground(brush)
        self.tableWidget.setItem(0, 2, item)
        item = QtWidgets.QTableWidgetItem()
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.BrushStyle.NoBrush)
        item.setForeground(brush)
        self.tableWidget.setItem(0, 3, item)
        item = QtWidgets.QTableWidgetItem()
        brush = QtGui.QBrush(QtGui.QColor(0, 255, 0))
        brush.setStyle(QtCore.Qt.BrushStyle.SolidPattern)
        item.setForeground(brush)
        self.tableWidget.setItem(0, 4, item)
        item = QtWidgets.QTableWidgetItem()
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.BrushStyle.NoBrush)
        item.setBackground(brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 0, 0))
        brush.setStyle(QtCore.Qt.BrushStyle.SolidPattern)
        item.setForeground(brush)
        self.tableWidget.setItem(0, 5, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setItem(1, 0, item)
        item = QtWidgets.QTableWidgetItem()
        brush = QtGui.QBrush(QtGui.QColor(200, 255, 255))
        brush.setStyle(QtCore.Qt.BrushStyle.SolidPattern)
        item.setBackground(brush)
        self.tableWidget.setItem(1, 1, item)
        item = QtWidgets.QTableWidgetItem()
        brush = QtGui.QBrush(QtGui.QColor(255, 127, 0))
        brush.setStyle(QtCore.Qt.BrushStyle.SolidPattern)
        item.setBackground(brush)
        self.tableWidget.setItem(1, 2, item)
        item = QtWidgets.QTableWidgetItem()
        brush = QtGui.QBrush(QtGui.QColor(200, 255, 255))
        brush.setStyle(QtCore.Qt.BrushStyle.NoBrush)
        item.setBackground(brush)
        self.tableWidget.setItem(1, 3, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setItem(1, 4, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setItem(1, 5, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setItem(2, 0, item)
        item = QtWidgets.QTableWidgetItem()
        brush = QtGui.QBrush(QtGui.QColor(200, 255, 255))
        brush.setStyle(QtCore.Qt.BrushStyle.SolidPattern)
        item.setBackground(brush)
        self.tableWidget.setItem(2, 1, item)
        item = QtWidgets.QTableWidgetItem()
        brush = QtGui.QBrush(QtGui.QColor(255, 127, 0))
        brush.setStyle(QtCore.Qt.BrushStyle.SolidPattern)
        item.setBackground(brush)
        self.tableWidget.setItem(2, 2, item)
        item = QtWidgets.QTableWidgetItem()
        brush = QtGui.QBrush(QtGui.QColor(255, 200, 229))
        brush.setStyle(QtCore.Qt.BrushStyle.SolidPattern)
        item.setBackground(brush)
        self.tableWidget.setItem(2, 3, item)
        item = QtWidgets.QTableWidgetItem()
        brush = QtGui.QBrush(QtGui.QColor(200, 200, 200))
        brush.setStyle(QtCore.Qt.BrushStyle.SolidPattern)
        item.setBackground(brush)
        self.tableWidget.setItem(2, 5, item)
        self.tableWidget.horizontalHeader().setVisible(False)
        self.tableWidget.horizontalHeader().setDefaultSectionSize(270)
        self.tableWidget.horizontalHeader().setHighlightSections(False)
        self.tableWidget.horizontalHeader().setMinimumSectionSize(270)
        self.tableWidget.verticalHeader().setVisible(False)
        self.tableWidget.verticalHeader().setDefaultSectionSize(25)
        self.tableWidget.verticalHeader().setMinimumSectionSize(25)
        self.horizontalLayout_8.addWidget(self.tableWidget)
        self.verticalLayout_5.addLayout(self.horizontalLayout_8)
        self.tb_channel = QtWidgets.QTableWidget(parent=self.tab)
        self.tb_channel.setMinimumSize(QtCore.QSize(1860, 692))
        self.tb_channel.setMaximumSize(QtCore.QSize(1860, 692))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.tb_channel.setFont(font)
        self.tb_channel.setRowCount(16)
        self.tb_channel.setColumnCount(8)
        self.tb_channel.setObjectName("tb_channel")
        self.tb_channel.horizontalHeader().setVisible(False)
        self.tb_channel.horizontalHeader().setDefaultSectionSize(232)
        self.tb_channel.horizontalHeader().setMinimumSectionSize(232)
        self.tb_channel.verticalHeader().setVisible(False)
        self.tb_channel.verticalHeader().setDefaultSectionSize(43)
        self.tb_channel.verticalHeader().setMinimumSectionSize(43)
        self.verticalLayout_5.addWidget(self.tb_channel)
        self.horizontalLayout_123.addLayout(self.verticalLayout_5)
        self.tabWidget.addTab(self.tab, "")
        self.CycTab = QtWidgets.QWidget()
        self.CycTab.setObjectName("CycTab")
        self.horizontalLayout_172 = QtWidgets.QHBoxLayout(self.CycTab)
        self.horizontalLayout_172.setObjectName("horizontalLayout_172")
        self.horizontalLayout_112 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_112.setObjectName("horizontalLayout_112")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout()
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.horizontalLayout_108 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_108.setObjectName("horizontalLayout_108")
        self.chk_cyclepath = QtWidgets.QCheckBox(parent=self.CycTab)
        self.chk_cyclepath.setMinimumSize(QtCore.QSize(240, 30))
        self.chk_cyclepath.setMaximumSize(QtCore.QSize(100, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.chk_cyclepath.setFont(font)
        self.chk_cyclepath.setChecked(True)
        self.chk_cyclepath.setObjectName("chk_cyclepath")
        self.horizontalLayout_108.addWidget(self.chk_cyclepath)
        self.chk_ectpath = QtWidgets.QCheckBox(parent=self.CycTab)
        self.chk_ectpath.setMinimumSize(QtCore.QSize(240, 30))
        self.chk_ectpath.setMaximumSize(QtCore.QSize(100, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.chk_ectpath.setFont(font)
        self.chk_ectpath.setChecked(False)
        self.chk_ectpath.setObjectName("chk_ectpath")
        self.horizontalLayout_108.addWidget(self.chk_ectpath)
        self.verticalLayout_6.addLayout(self.horizontalLayout_108)
        self.horizontalLayout_119 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_119.setObjectName("horizontalLayout_119")
        self.stepnum_2 = QtWidgets.QPlainTextEdit(parent=self.CycTab)
        self.stepnum_2.setMinimumSize(QtCore.QSize(240, 70))
        self.stepnum_2.setMaximumSize(QtCore.QSize(240, 70))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.stepnum_2.setFont(font)
        self.stepnum_2.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhNone)
        self.stepnum_2.setPlainText("")
        self.stepnum_2.setObjectName("stepnum_2")
        self.horizontalLayout_119.addWidget(self.stepnum_2)
        self.cycle_tab_reset = QtWidgets.QPushButton(parent=self.CycTab)
        self.cycle_tab_reset.setMinimumSize(QtCore.QSize(240, 70))
        self.cycle_tab_reset.setMaximumSize(QtCore.QSize(240, 70))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(True)
        font.setUnderline(True)
        font.setWeight(75)
        self.cycle_tab_reset.setFont(font)
        self.cycle_tab_reset.setObjectName("cycle_tab_reset")
        self.horizontalLayout_119.addWidget(self.cycle_tab_reset)
        self.verticalLayout_6.addLayout(self.horizontalLayout_119)
        self.capacitygroup = QtWidgets.QGroupBox(parent=self.CycTab)
        self.capacitygroup.setMinimumSize(QtCore.QSize(502, 120))
        self.capacitygroup.setMaximumSize(QtCore.QSize(502, 120))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.capacitygroup.setFont(font)
        self.capacitygroup.setObjectName("capacitygroup")
        self.horizontalLayout_122 = QtWidgets.QHBoxLayout(self.capacitygroup)
        self.horizontalLayout_122.setObjectName("horizontalLayout_122")
        self.verticalLayout_18 = QtWidgets.QVBoxLayout()
        self.verticalLayout_18.setObjectName("verticalLayout_18")
        self.horizontalLayout_120 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_120.setObjectName("horizontalLayout_120")
        self.inicaprate = QtWidgets.QRadioButton(parent=self.capacitygroup)
        self.inicaprate.setMinimumSize(QtCore.QSize(352, 50))
        self.inicaprate.setMaximumSize(QtCore.QSize(352, 50))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.inicaprate.setFont(font)
        self.inicaprate.setChecked(True)
        self.inicaprate.setObjectName("inicaprate")
        self.horizontalLayout_120.addWidget(self.inicaprate)
        self.ratetext = QtWidgets.QLineEdit(parent=self.capacitygroup)
        self.ratetext.setMinimumSize(QtCore.QSize(120, 25))
        self.ratetext.setMaximumSize(QtCore.QSize(120, 25))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.ratetext.setFont(font)
        self.ratetext.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhFormattedNumbersOnly)
        self.ratetext.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.ratetext.setObjectName("ratetext")
        self.horizontalLayout_120.addWidget(self.ratetext)
        self.verticalLayout_18.addLayout(self.horizontalLayout_120)
        self.horizontalLayout_121 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_121.setObjectName("horizontalLayout_121")
        self.inicaptype = QtWidgets.QRadioButton(parent=self.capacitygroup)
        self.inicaptype.setMinimumSize(QtCore.QSize(352, 20))
        self.inicaptype.setMaximumSize(QtCore.QSize(352, 20))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.inicaptype.setFont(font)
        self.inicaptype.setChecked(False)
        self.inicaptype.setObjectName("inicaptype")
        self.horizontalLayout_121.addWidget(self.inicaptype)
        self.capacitytext = QtWidgets.QLineEdit(parent=self.capacitygroup)
        self.capacitytext.setMinimumSize(QtCore.QSize(120, 25))
        self.capacitytext.setMaximumSize(QtCore.QSize(120, 25))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.capacitytext.setFont(font)
        self.capacitytext.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhFormattedNumbersOnly)
        self.capacitytext.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.capacitytext.setObjectName("capacitytext")
        self.horizontalLayout_121.addWidget(self.capacitytext)
        self.verticalLayout_18.addLayout(self.horizontalLayout_121)
        self.horizontalLayout_122.addLayout(self.verticalLayout_18)
        self.verticalLayout_6.addWidget(self.capacitygroup)
        self.tabWidget_2 = QtWidgets.QTabWidget(parent=self.CycTab)
        self.tabWidget_2.setMinimumSize(QtCore.QSize(502, 594))
        self.tabWidget_2.setMaximumSize(QtCore.QSize(502, 594))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.tabWidget_2.setFont(font)
        self.tabWidget_2.setObjectName("tabWidget_2")
        self.tab_5 = QtWidgets.QWidget()
        self.tab_5.setObjectName("tab_5")
        self.horizontalLayout_107 = QtWidgets.QHBoxLayout(self.tab_5)
        self.horizontalLayout_107.setObjectName("horizontalLayout_107")
        self.verticalLayout_17 = QtWidgets.QVBoxLayout()
        self.verticalLayout_17.setObjectName("verticalLayout_17")
        self.verticalLayout_14 = QtWidgets.QVBoxLayout()
        self.verticalLayout_14.setObjectName("verticalLayout_14")
        self.dcirchk = QtWidgets.QRadioButton(parent=self.tab_5)
        self.dcirchk.setMinimumSize(QtCore.QSize(450, 30))
        self.dcirchk.setMaximumSize(QtCore.QSize(450, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.dcirchk.setFont(font)
        self.dcirchk.setObjectName("dcirchk")
        self.verticalLayout_14.addWidget(self.dcirchk)
        self.pulsedcir = QtWidgets.QRadioButton(parent=self.tab_5)
        self.pulsedcir.setMinimumSize(QtCore.QSize(450, 30))
        self.pulsedcir.setMaximumSize(QtCore.QSize(450, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.pulsedcir.setFont(font)
        self.pulsedcir.setChecked(False)
        self.pulsedcir.setObjectName("pulsedcir")
        self.verticalLayout_14.addWidget(self.pulsedcir)
        self.mkdcir = QtWidgets.QRadioButton(parent=self.tab_5)
        self.mkdcir.setMinimumSize(QtCore.QSize(450, 30))
        self.mkdcir.setMaximumSize(QtCore.QSize(450, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.mkdcir.setFont(font)
        self.mkdcir.setCheckable(True)
        self.mkdcir.setChecked(True)
        self.mkdcir.setObjectName("mkdcir")
        self.verticalLayout_14.addWidget(self.mkdcir)
        self.dcirchk_2 = QtWidgets.QCheckBox(parent=self.tab_5)
        self.dcirchk_2.setMinimumSize(QtCore.QSize(234, 30))
        self.dcirchk_2.setMaximumSize(QtCore.QSize(234, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.dcirchk_2.setFont(font)
        self.dcirchk_2.setObjectName("dcirchk_2")
        self.verticalLayout_14.addWidget(self.dcirchk_2)
        self.verticalLayout_17.addLayout(self.verticalLayout_14)
        self.horizontalLayout_86 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_86.setObjectName("horizontalLayout_86")
        self.cycxlabel_2 = QtWidgets.QLabel(parent=self.tab_5)
        self.cycxlabel_2.setMinimumSize(QtCore.QSize(215, 30))
        self.cycxlabel_2.setMaximumSize(QtCore.QSize(215, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.cycxlabel_2.setFont(font)
        self.cycxlabel_2.setObjectName("cycxlabel_2")
        self.horizontalLayout_86.addWidget(self.cycxlabel_2)
        self.tcyclerngyhl = QtWidgets.QLineEdit(parent=self.tab_5)
        self.tcyclerngyhl.setMinimumSize(QtCore.QSize(215, 30))
        self.tcyclerngyhl.setMaximumSize(QtCore.QSize(215, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.tcyclerngyhl.setFont(font)
        self.tcyclerngyhl.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhFormattedNumbersOnly)
        self.tcyclerngyhl.setClearButtonEnabled(False)
        self.tcyclerngyhl.setObjectName("tcyclerngyhl")
        self.horizontalLayout_86.addWidget(self.tcyclerngyhl)
        self.verticalLayout_17.addLayout(self.horizontalLayout_86)
        self.horizontalLayout_87 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_87.setObjectName("horizontalLayout_87")
        self.cycxlabel_3 = QtWidgets.QLabel(parent=self.tab_5)
        self.cycxlabel_3.setMinimumSize(QtCore.QSize(215, 30))
        self.cycxlabel_3.setMaximumSize(QtCore.QSize(215, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.cycxlabel_3.setFont(font)
        self.cycxlabel_3.setObjectName("cycxlabel_3")
        self.horizontalLayout_87.addWidget(self.cycxlabel_3)
        self.tcyclerngyll = QtWidgets.QLineEdit(parent=self.tab_5)
        self.tcyclerngyll.setMinimumSize(QtCore.QSize(215, 30))
        self.tcyclerngyll.setMaximumSize(QtCore.QSize(215, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.tcyclerngyll.setFont(font)
        self.tcyclerngyll.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhFormattedNumbersOnly)
        self.tcyclerngyll.setObjectName("tcyclerngyll")
        self.horizontalLayout_87.addWidget(self.tcyclerngyll)
        self.verticalLayout_17.addLayout(self.horizontalLayout_87)
        self.horizontalLayout_88 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_88.setObjectName("horizontalLayout_88")
        self.cycxlabel = QtWidgets.QLabel(parent=self.tab_5)
        self.cycxlabel.setMinimumSize(QtCore.QSize(215, 30))
        self.cycxlabel.setMaximumSize(QtCore.QSize(215, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.cycxlabel.setFont(font)
        self.cycxlabel.setObjectName("cycxlabel")
        self.horizontalLayout_88.addWidget(self.cycxlabel)
        self.tcyclerng = QtWidgets.QLineEdit(parent=self.tab_5)
        self.tcyclerng.setMinimumSize(QtCore.QSize(215, 30))
        self.tcyclerng.setMaximumSize(QtCore.QSize(215, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.tcyclerng.setFont(font)
        self.tcyclerng.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhDigitsOnly)
        self.tcyclerng.setObjectName("tcyclerng")
        self.horizontalLayout_88.addWidget(self.tcyclerng)
        self.verticalLayout_17.addLayout(self.horizontalLayout_88)
        self.horizontalLayout_89 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_89.setObjectName("horizontalLayout_89")
        self.dcirscalelb = QtWidgets.QLabel(parent=self.tab_5)
        self.dcirscalelb.setMinimumSize(QtCore.QSize(215, 30))
        self.dcirscalelb.setMaximumSize(QtCore.QSize(215, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.dcirscalelb.setFont(font)
        self.dcirscalelb.setObjectName("dcirscalelb")
        self.horizontalLayout_89.addWidget(self.dcirscalelb)
        self.dcirscale = QtWidgets.QLineEdit(parent=self.tab_5)
        self.dcirscale.setMinimumSize(QtCore.QSize(215, 30))
        self.dcirscale.setMaximumSize(QtCore.QSize(215, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.dcirscale.setFont(font)
        self.dcirscale.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhDigitsOnly)
        self.dcirscale.setObjectName("dcirscale")
        self.horizontalLayout_89.addWidget(self.dcirscale)
        self.verticalLayout_17.addLayout(self.horizontalLayout_89)
        self.horizontalLayout_90 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_90.setObjectName("horizontalLayout_90")
        self.indiv_cycle = QtWidgets.QPushButton(parent=self.tab_5)
        self.indiv_cycle.setMinimumSize(QtCore.QSize(215, 70))
        self.indiv_cycle.setMaximumSize(QtCore.QSize(215, 70))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.indiv_cycle.setFont(font)
        self.indiv_cycle.setObjectName("indiv_cycle")
        self.horizontalLayout_90.addWidget(self.indiv_cycle)
        self.overall_cycle = QtWidgets.QPushButton(parent=self.tab_5)
        self.overall_cycle.setMinimumSize(QtCore.QSize(215, 70))
        self.overall_cycle.setMaximumSize(QtCore.QSize(215, 70))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.overall_cycle.setFont(font)
        self.overall_cycle.setObjectName("overall_cycle")
        self.horizontalLayout_90.addWidget(self.overall_cycle)
        self.verticalLayout_17.addLayout(self.horizontalLayout_90)
        self.horizontalLayout_91 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_91.setObjectName("horizontalLayout_91")
        self.link_cycle = QtWidgets.QPushButton(parent=self.tab_5)
        self.link_cycle.setMinimumSize(QtCore.QSize(215, 70))
        self.link_cycle.setMaximumSize(QtCore.QSize(215, 70))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        font.setKerning(True)
        self.link_cycle.setFont(font)
        self.link_cycle.setObjectName("link_cycle")
        self.horizontalLayout_91.addWidget(self.link_cycle)
        self.AppCycConfirm = QtWidgets.QPushButton(parent=self.tab_5)
        self.AppCycConfirm.setMinimumSize(QtCore.QSize(215, 70))
        self.AppCycConfirm.setMaximumSize(QtCore.QSize(215, 70))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.AppCycConfirm.setFont(font)
        self.AppCycConfirm.setObjectName("AppCycConfirm")
        self.horizontalLayout_91.addWidget(self.AppCycConfirm)
        self.verticalLayout_17.addLayout(self.horizontalLayout_91)
        self.horizontalLayout_92 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_92.setObjectName("horizontalLayout_92")
        self.link_cycle_indiv = QtWidgets.QPushButton(parent=self.tab_5)
        self.link_cycle_indiv.setMinimumSize(QtCore.QSize(215, 70))
        self.link_cycle_indiv.setMaximumSize(QtCore.QSize(215, 70))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        font.setKerning(True)
        self.link_cycle_indiv.setFont(font)
        self.link_cycle_indiv.setObjectName("link_cycle_indiv")
        self.horizontalLayout_92.addWidget(self.link_cycle_indiv)
        self.link_cycle_overall = QtWidgets.QPushButton(parent=self.tab_5)
        self.link_cycle_overall.setMinimumSize(QtCore.QSize(215, 70))
        self.link_cycle_overall.setMaximumSize(QtCore.QSize(215, 70))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        font.setKerning(True)
        self.link_cycle_overall.setFont(font)
        self.link_cycle_overall.setObjectName("link_cycle_overall")
        self.horizontalLayout_92.addWidget(self.link_cycle_overall)
        self.verticalLayout_17.addLayout(self.horizontalLayout_92)
        self.horizontalLayout_107.addLayout(self.verticalLayout_17)
        self.tabWidget_2.addTab(self.tab_5, "")
        self.tab_6 = QtWidgets.QWidget()
        self.tab_6.setObjectName("tab_6")
        self.horizontalLayout_17 = QtWidgets.QHBoxLayout(self.tab_6)
        self.horizontalLayout_17.setObjectName("horizontalLayout_17")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.horizontalLayout_15 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_15.setObjectName("horizontalLayout_15")
        self.CycProfile = QtWidgets.QRadioButton(parent=self.tab_6)
        self.CycProfile.setMinimumSize(QtCore.QSize(0, 30))
        self.CycProfile.setMaximumSize(QtCore.QSize(300, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.CycProfile.setFont(font)
        self.CycProfile.setChecked(True)
        self.CycProfile.setObjectName("CycProfile")
        self.horizontalLayout_15.addWidget(self.CycProfile)
        self.CellProfile = QtWidgets.QRadioButton(parent=self.tab_6)
        self.CellProfile.setMinimumSize(QtCore.QSize(0, 30))
        self.CellProfile.setMaximumSize(QtCore.QSize(300, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.CellProfile.setFont(font)
        self.CellProfile.setObjectName("CellProfile")
        self.horizontalLayout_15.addWidget(self.CellProfile)
        self.chk_dqdv = QtWidgets.QCheckBox(parent=self.tab_6)
        self.chk_dqdv.setMinimumSize(QtCore.QSize(120, 30))
        self.chk_dqdv.setMaximumSize(QtCore.QSize(120, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.chk_dqdv.setFont(font)
        self.chk_dqdv.setChecked(False)
        self.chk_dqdv.setObjectName("chk_dqdv")
        self.horizontalLayout_15.addWidget(self.chk_dqdv)
        self.verticalLayout_4.addLayout(self.horizontalLayout_15)
        self.horizontalLayout_111 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_111.setObjectName("horizontalLayout_111")
        self.stepnumlb = QtWidgets.QLabel(parent=self.tab_6)
        self.stepnumlb.setMinimumSize(QtCore.QSize(0, 60))
        self.stepnumlb.setMaximumSize(QtCore.QSize(215, 60))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.stepnumlb.setFont(font)
        self.stepnumlb.setObjectName("stepnumlb")
        self.horizontalLayout_111.addWidget(self.stepnumlb)
        self.stepnum = QtWidgets.QPlainTextEdit(parent=self.tab_6)
        self.stepnum.setMinimumSize(QtCore.QSize(0, 60))
        self.stepnum.setMaximumSize(QtCore.QSize(215, 60))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.stepnum.setFont(font)
        self.stepnum.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhNone)
        self.stepnum.setObjectName("stepnum")
        self.horizontalLayout_111.addWidget(self.stepnum)
        self.verticalLayout_4.addLayout(self.horizontalLayout_111)
        self.horizontalLayout_85 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_85.setObjectName("horizontalLayout_85")
        self.smoothlb_3 = QtWidgets.QLabel(parent=self.tab_6)
        self.smoothlb_3.setMinimumSize(QtCore.QSize(215, 25))
        self.smoothlb_3.setMaximumSize(QtCore.QSize(215, 25))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.smoothlb_3.setFont(font)
        self.smoothlb_3.setObjectName("smoothlb_3")
        self.horizontalLayout_85.addWidget(self.smoothlb_3)
        self.volrngyhl = QtWidgets.QLineEdit(parent=self.tab_6)
        self.volrngyhl.setMinimumSize(QtCore.QSize(215, 25))
        self.volrngyhl.setMaximumSize(QtCore.QSize(215, 25))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.volrngyhl.setFont(font)
        self.volrngyhl.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhDigitsOnly)
        self.volrngyhl.setObjectName("volrngyhl")
        self.horizontalLayout_85.addWidget(self.volrngyhl)
        self.verticalLayout_4.addLayout(self.horizontalLayout_85)
        self.horizontalLayout_110 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_110.setObjectName("horizontalLayout_110")
        self.smoothlb_2 = QtWidgets.QLabel(parent=self.tab_6)
        self.smoothlb_2.setMinimumSize(QtCore.QSize(215, 25))
        self.smoothlb_2.setMaximumSize(QtCore.QSize(215, 25))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.smoothlb_2.setFont(font)
        self.smoothlb_2.setObjectName("smoothlb_2")
        self.horizontalLayout_110.addWidget(self.smoothlb_2)
        self.volrngyll = QtWidgets.QLineEdit(parent=self.tab_6)
        self.volrngyll.setMinimumSize(QtCore.QSize(215, 25))
        self.volrngyll.setMaximumSize(QtCore.QSize(215, 25))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.volrngyll.setFont(font)
        self.volrngyll.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhDigitsOnly)
        self.volrngyll.setObjectName("volrngyll")
        self.horizontalLayout_110.addWidget(self.volrngyll)
        self.verticalLayout_4.addLayout(self.horizontalLayout_110)
        self.horizontalLayout_109 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_109.setObjectName("horizontalLayout_109")
        self.smoothlb_4 = QtWidgets.QLabel(parent=self.tab_6)
        self.smoothlb_4.setMinimumSize(QtCore.QSize(215, 25))
        self.smoothlb_4.setMaximumSize(QtCore.QSize(215, 25))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.smoothlb_4.setFont(font)
        self.smoothlb_4.setObjectName("smoothlb_4")
        self.horizontalLayout_109.addWidget(self.smoothlb_4)
        self.volrnggap = QtWidgets.QLineEdit(parent=self.tab_6)
        self.volrnggap.setMinimumSize(QtCore.QSize(215, 25))
        self.volrnggap.setMaximumSize(QtCore.QSize(215, 25))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.volrnggap.setFont(font)
        self.volrnggap.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhDigitsOnly)
        self.volrnggap.setObjectName("volrnggap")
        self.horizontalLayout_109.addWidget(self.volrnggap)
        self.verticalLayout_4.addLayout(self.horizontalLayout_109)
        self.horizontalLayout_113 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_113.setObjectName("horizontalLayout_113")
        self.smoothlb = QtWidgets.QLabel(parent=self.tab_6)
        self.smoothlb.setMinimumSize(QtCore.QSize(215, 25))
        self.smoothlb.setMaximumSize(QtCore.QSize(215, 25))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.smoothlb.setFont(font)
        self.smoothlb.setObjectName("smoothlb")
        self.horizontalLayout_113.addWidget(self.smoothlb)
        self.smooth = QtWidgets.QLineEdit(parent=self.tab_6)
        self.smooth.setMinimumSize(QtCore.QSize(215, 25))
        self.smooth.setMaximumSize(QtCore.QSize(215, 25))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.smooth.setFont(font)
        self.smooth.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhDigitsOnly)
        self.smooth.setObjectName("smooth")
        self.horizontalLayout_113.addWidget(self.smooth)
        self.verticalLayout_4.addLayout(self.horizontalLayout_113)
        self.horizontalLayout_114 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_114.setObjectName("horizontalLayout_114")
        self.cutofflb = QtWidgets.QLabel(parent=self.tab_6)
        self.cutofflb.setMinimumSize(QtCore.QSize(215, 25))
        self.cutofflb.setMaximumSize(QtCore.QSize(215, 25))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.cutofflb.setFont(font)
        self.cutofflb.setObjectName("cutofflb")
        self.horizontalLayout_114.addWidget(self.cutofflb)
        self.cutoff = QtWidgets.QLineEdit(parent=self.tab_6)
        self.cutoff.setMinimumSize(QtCore.QSize(215, 25))
        self.cutoff.setMaximumSize(QtCore.QSize(215, 25))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.cutoff.setFont(font)
        self.cutoff.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhFormattedNumbersOnly)
        self.cutoff.setObjectName("cutoff")
        self.horizontalLayout_114.addWidget(self.cutoff)
        self.verticalLayout_4.addLayout(self.horizontalLayout_114)
        self.horizontalLayout_115 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_115.setObjectName("horizontalLayout_115")
        self.dqdvscalelb = QtWidgets.QLabel(parent=self.tab_6)
        self.dqdvscalelb.setMinimumSize(QtCore.QSize(215, 25))
        self.dqdvscalelb.setMaximumSize(QtCore.QSize(215, 25))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.dqdvscalelb.setFont(font)
        self.dqdvscalelb.setObjectName("dqdvscalelb")
        self.horizontalLayout_115.addWidget(self.dqdvscalelb)
        self.dqdvscale = QtWidgets.QLineEdit(parent=self.tab_6)
        self.dqdvscale.setMinimumSize(QtCore.QSize(215, 25))
        self.dqdvscale.setMaximumSize(QtCore.QSize(215, 25))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.dqdvscale.setFont(font)
        self.dqdvscale.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhDigitsOnly)
        self.dqdvscale.setObjectName("dqdvscale")
        self.horizontalLayout_115.addWidget(self.dqdvscale)
        self.verticalLayout_4.addLayout(self.horizontalLayout_115)
        self.horizontalLayout_116 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_116.setObjectName("horizontalLayout_116")
        self.StepConfirm = QtWidgets.QPushButton(parent=self.tab_6)
        self.StepConfirm.setMinimumSize(QtCore.QSize(215, 70))
        self.StepConfirm.setMaximumSize(QtCore.QSize(215, 70))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.StepConfirm.setFont(font)
        self.StepConfirm.setObjectName("StepConfirm")
        self.horizontalLayout_116.addWidget(self.StepConfirm)
        self.ChgConfirm = QtWidgets.QPushButton(parent=self.tab_6)
        self.ChgConfirm.setMinimumSize(QtCore.QSize(215, 70))
        self.ChgConfirm.setMaximumSize(QtCore.QSize(215, 70))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.ChgConfirm.setFont(font)
        self.ChgConfirm.setObjectName("ChgConfirm")
        self.horizontalLayout_116.addWidget(self.ChgConfirm)
        self.verticalLayout_4.addLayout(self.horizontalLayout_116)
        self.horizontalLayout_117 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_117.setObjectName("horizontalLayout_117")
        self.RateConfirm = QtWidgets.QPushButton(parent=self.tab_6)
        self.RateConfirm.setMinimumSize(QtCore.QSize(215, 70))
        self.RateConfirm.setMaximumSize(QtCore.QSize(215, 70))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.RateConfirm.setFont(font)
        self.RateConfirm.setObjectName("RateConfirm")
        self.horizontalLayout_117.addWidget(self.RateConfirm)
        self.DchgConfirm = QtWidgets.QPushButton(parent=self.tab_6)
        self.DchgConfirm.setMinimumSize(QtCore.QSize(215, 70))
        self.DchgConfirm.setMaximumSize(QtCore.QSize(215, 70))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.DchgConfirm.setFont(font)
        self.DchgConfirm.setObjectName("DchgConfirm")
        self.horizontalLayout_117.addWidget(self.DchgConfirm)
        self.verticalLayout_4.addLayout(self.horizontalLayout_117)
        self.horizontalLayout_118 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_118.setObjectName("horizontalLayout_118")
        self.ContinueConfirm = QtWidgets.QPushButton(parent=self.tab_6)
        self.ContinueConfirm.setMinimumSize(QtCore.QSize(215, 70))
        self.ContinueConfirm.setMaximumSize(QtCore.QSize(215, 70))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.ContinueConfirm.setFont(font)
        self.ContinueConfirm.setObjectName("ContinueConfirm")
        self.horizontalLayout_118.addWidget(self.ContinueConfirm)
        self.DCIRConfirm = QtWidgets.QPushButton(parent=self.tab_6)
        self.DCIRConfirm.setMinimumSize(QtCore.QSize(215, 70))
        self.DCIRConfirm.setMaximumSize(QtCore.QSize(215, 70))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.DCIRConfirm.setFont(font)
        self.DCIRConfirm.setObjectName("DCIRConfirm")
        self.horizontalLayout_118.addWidget(self.DCIRConfirm)
        self.verticalLayout_4.addLayout(self.horizontalLayout_118)
        self.horizontalLayout_17.addLayout(self.verticalLayout_4)
        self.tabWidget_2.addTab(self.tab_6, "")
        self.verticalLayout_6.addWidget(self.tabWidget_2)
        self.horizontalLayout_112.addLayout(self.verticalLayout_6)
        self.cycle_tab = QtWidgets.QTabWidget(parent=self.CycTab)
        self.cycle_tab.setMinimumSize(QtCore.QSize(1350, 830))
        self.cycle_tab.setMaximumSize(QtCore.QSize(1350, 830))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.cycle_tab.setFont(font)
        self.cycle_tab.setObjectName("cycle_tab")
        self.horizontalLayout_112.addWidget(self.cycle_tab)
        self.horizontalLayout_172.addLayout(self.horizontalLayout_112)
        self.tabWidget.addTab(self.CycTab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.horizontalLayout_135 = QtWidgets.QHBoxLayout(self.tab_2)
        self.horizontalLayout_135.setObjectName("horizontalLayout_135")
        self.horizontalLayout_134 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_134.setObjectName("horizontalLayout_134")
        self.verticalLayout_9 = QtWidgets.QVBoxLayout()
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.verticalLayout_8 = QtWidgets.QVBoxLayout()
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.horizontalLayout_129 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_129.setObjectName("horizontalLayout_129")
        self.label_10 = QtWidgets.QLabel(parent=self.tab_2)
        self.label_10.setMinimumSize(QtCore.QSize(369, 30))
        self.label_10.setMaximumSize(QtCore.QSize(369, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.label_10.setFont(font)
        self.label_10.setObjectName("label_10")
        self.horizontalLayout_129.addWidget(self.label_10)
        spacerItem1 = QtWidgets.QSpacerItem(201, 30, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.horizontalLayout_129.addItem(spacerItem1)
        self.chk_coincell = QtWidgets.QCheckBox(parent=self.tab_2)
        self.chk_coincell.setMinimumSize(QtCore.QSize(70, 18))
        self.chk_coincell.setMaximumSize(QtCore.QSize(70, 18))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.chk_coincell.setFont(font)
        self.chk_coincell.setObjectName("chk_coincell")
        self.horizontalLayout_129.addWidget(self.chk_coincell)
        self.verticalLayout_8.addLayout(self.horizontalLayout_129)
        self.horizontalLayout_14 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_14.setObjectName("horizontalLayout_14")
        self.cycxlabel_7 = QtWidgets.QLabel(parent=self.tab_2)
        self.cycxlabel_7.setMinimumSize(QtCore.QSize(526, 30))
        self.cycxlabel_7.setMaximumSize(QtCore.QSize(526, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.cycxlabel_7.setFont(font)
        self.cycxlabel_7.setObjectName("cycxlabel_7")
        self.horizontalLayout_14.addWidget(self.cycxlabel_7)
        self.ptn_load = QtWidgets.QPushButton(parent=self.tab_2)
        self.ptn_load.setMinimumSize(QtCore.QSize(120, 50))
        self.ptn_load.setMaximumSize(QtCore.QSize(120, 50))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.ptn_load.setFont(font)
        self.ptn_load.setObjectName("ptn_load")
        self.horizontalLayout_14.addWidget(self.ptn_load)
        self.verticalLayout_8.addLayout(self.horizontalLayout_14)
        self.ptn_ori_path = QtWidgets.QLineEdit(parent=self.tab_2)
        self.ptn_ori_path.setMinimumSize(QtCore.QSize(640, 30))
        self.ptn_ori_path.setMaximumSize(QtCore.QSize(640, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.ptn_ori_path.setFont(font)
        self.ptn_ori_path.setObjectName("ptn_ori_path")
        self.verticalLayout_8.addWidget(self.ptn_ori_path)
        self.line_16 = QtWidgets.QFrame(parent=self.tab_2)
        self.line_16.setMinimumSize(QtCore.QSize(654, 3))
        self.line_16.setMaximumSize(QtCore.QSize(654, 3))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.line_16.setFont(font)
        self.line_16.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        self.line_16.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.line_16.setObjectName("line_16")
        self.verticalLayout_8.addWidget(self.line_16)
        self.verticalLayout_9.addLayout(self.verticalLayout_8)
        self.horizontalLayout_133 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_133.setObjectName("horizontalLayout_133")
        self.verticalLayout_69 = QtWidgets.QVBoxLayout()
        self.verticalLayout_69.setObjectName("verticalLayout_69")
        self.groupBox_8 = QtWidgets.QGroupBox(parent=self.tab_2)
        self.groupBox_8.setMinimumSize(QtCore.QSize(140, 300))
        self.groupBox_8.setMaximumSize(QtCore.QSize(140, 300))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.groupBox_8.setFont(font)
        self.groupBox_8.setObjectName("groupBox_8")
        self.gridLayout = QtWidgets.QGridLayout(self.groupBox_8)
        self.gridLayout.setObjectName("gridLayout")
        self.verticalLayout_34 = QtWidgets.QVBoxLayout()
        self.verticalLayout_34.setObjectName("verticalLayout_34")
        self.cycxlabel_5 = QtWidgets.QLabel(parent=self.groupBox_8)
        self.cycxlabel_5.setMinimumSize(QtCore.QSize(125, 20))
        self.cycxlabel_5.setMaximumSize(QtCore.QSize(125, 20))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.cycxlabel_5.setFont(font)
        self.cycxlabel_5.setObjectName("cycxlabel_5")
        self.verticalLayout_34.addWidget(self.cycxlabel_5)
        self.ptn_crate = QtWidgets.QLineEdit(parent=self.groupBox_8)
        self.ptn_crate.setMinimumSize(QtCore.QSize(125, 30))
        self.ptn_crate.setMaximumSize(QtCore.QSize(125, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.ptn_crate.setFont(font)
        self.ptn_crate.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhFormattedNumbersOnly)
        self.ptn_crate.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.ptn_crate.setObjectName("ptn_crate")
        self.verticalLayout_34.addWidget(self.ptn_crate)
        self.cycxlabel_6 = QtWidgets.QLabel(parent=self.groupBox_8)
        self.cycxlabel_6.setMinimumSize(QtCore.QSize(125, 20))
        self.cycxlabel_6.setMaximumSize(QtCore.QSize(125, 20))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.cycxlabel_6.setFont(font)
        self.cycxlabel_6.setObjectName("cycxlabel_6")
        self.verticalLayout_34.addWidget(self.cycxlabel_6)
        self.ptn_capacity = QtWidgets.QLineEdit(parent=self.groupBox_8)
        self.ptn_capacity.setMinimumSize(QtCore.QSize(125, 30))
        self.ptn_capacity.setMaximumSize(QtCore.QSize(125, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.ptn_capacity.setFont(font)
        self.ptn_capacity.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhFormattedNumbersOnly)
        self.ptn_capacity.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.ptn_capacity.setObjectName("ptn_capacity")
        self.verticalLayout_34.addWidget(self.ptn_capacity)
        spacerItem2 = QtWidgets.QSpacerItem(118, 50, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Fixed)
        self.verticalLayout_34.addItem(spacerItem2)
        self.chg_ptn = QtWidgets.QPushButton(parent=self.groupBox_8)
        self.chg_ptn.setMinimumSize(QtCore.QSize(125, 50))
        self.chg_ptn.setMaximumSize(QtCore.QSize(125, 50))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.chg_ptn.setFont(font)
        self.chg_ptn.setObjectName("chg_ptn")
        self.verticalLayout_34.addWidget(self.chg_ptn)
        self.gridLayout.addLayout(self.verticalLayout_34, 0, 0, 1, 1)
        self.verticalLayout_69.addWidget(self.groupBox_8)
        self.groupBox_5 = QtWidgets.QGroupBox(parent=self.tab_2)
        self.groupBox_5.setMinimumSize(QtCore.QSize(140, 300))
        self.groupBox_5.setMaximumSize(QtCore.QSize(140, 300))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.groupBox_5.setFont(font)
        self.groupBox_5.setObjectName("groupBox_5")
        self.gridLayout_17 = QtWidgets.QGridLayout(self.groupBox_5)
        self.gridLayout_17.setObjectName("gridLayout_17")
        self.verticalLayout_31 = QtWidgets.QVBoxLayout()
        self.verticalLayout_31.setObjectName("verticalLayout_31")
        self.cycxlabel_19 = QtWidgets.QLabel(parent=self.groupBox_5)
        self.cycxlabel_19.setMinimumSize(QtCore.QSize(125, 20))
        self.cycxlabel_19.setMaximumSize(QtCore.QSize(125, 20))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.cycxlabel_19.setFont(font)
        self.cycxlabel_19.setObjectName("cycxlabel_19")
        self.verticalLayout_31.addWidget(self.cycxlabel_19)
        self.ptn_chgv_pre = QtWidgets.QLineEdit(parent=self.groupBox_5)
        self.ptn_chgv_pre.setMinimumSize(QtCore.QSize(125, 30))
        self.ptn_chgv_pre.setMaximumSize(QtCore.QSize(125, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.ptn_chgv_pre.setFont(font)
        self.ptn_chgv_pre.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhFormattedNumbersOnly)
        self.ptn_chgv_pre.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.ptn_chgv_pre.setObjectName("ptn_chgv_pre")
        self.verticalLayout_31.addWidget(self.ptn_chgv_pre)
        self.cycxlabel_20 = QtWidgets.QLabel(parent=self.groupBox_5)
        self.cycxlabel_20.setMinimumSize(QtCore.QSize(125, 20))
        self.cycxlabel_20.setMaximumSize(QtCore.QSize(125, 20))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.cycxlabel_20.setFont(font)
        self.cycxlabel_20.setObjectName("cycxlabel_20")
        self.verticalLayout_31.addWidget(self.cycxlabel_20)
        self.ptn_chgv_after = QtWidgets.QLineEdit(parent=self.groupBox_5)
        self.ptn_chgv_after.setMinimumSize(QtCore.QSize(125, 30))
        self.ptn_chgv_after.setMaximumSize(QtCore.QSize(125, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.ptn_chgv_after.setFont(font)
        self.ptn_chgv_after.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhFormattedNumbersOnly)
        self.ptn_chgv_after.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.ptn_chgv_after.setObjectName("ptn_chgv_after")
        self.verticalLayout_31.addWidget(self.ptn_chgv_after)
        spacerItem3 = QtWidgets.QSpacerItem(118, 50, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Fixed)
        self.verticalLayout_31.addItem(spacerItem3)
        self.chg_ptn_chgv = QtWidgets.QPushButton(parent=self.groupBox_5)
        self.chg_ptn_chgv.setMinimumSize(QtCore.QSize(125, 50))
        self.chg_ptn_chgv.setMaximumSize(QtCore.QSize(125, 50))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.chg_ptn_chgv.setFont(font)
        self.chg_ptn_chgv.setObjectName("chg_ptn_chgv")
        self.verticalLayout_31.addWidget(self.chg_ptn_chgv)
        self.gridLayout_17.addLayout(self.verticalLayout_31, 0, 0, 1, 1)
        self.verticalLayout_69.addWidget(self.groupBox_5)
        self.horizontalLayout_133.addLayout(self.verticalLayout_69)
        spacerItem4 = QtWidgets.QSpacerItem(17, 608, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.horizontalLayout_133.addItem(spacerItem4)
        self.verticalLayout_70 = QtWidgets.QVBoxLayout()
        self.verticalLayout_70.setObjectName("verticalLayout_70")
        self.groupBox_2 = QtWidgets.QGroupBox(parent=self.tab_2)
        self.groupBox_2.setMinimumSize(QtCore.QSize(140, 300))
        self.groupBox_2.setMaximumSize(QtCore.QSize(140, 300))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.groupBox_2.setFont(font)
        self.groupBox_2.setObjectName("groupBox_2")
        self.gridLayout_12 = QtWidgets.QGridLayout(self.groupBox_2)
        self.gridLayout_12.setObjectName("gridLayout_12")
        self.verticalLayout_21 = QtWidgets.QVBoxLayout()
        self.verticalLayout_21.setObjectName("verticalLayout_21")
        self.cycxlabel_9 = QtWidgets.QLabel(parent=self.groupBox_2)
        self.cycxlabel_9.setMinimumSize(QtCore.QSize(125, 20))
        self.cycxlabel_9.setMaximumSize(QtCore.QSize(125, 20))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.cycxlabel_9.setFont(font)
        self.cycxlabel_9.setObjectName("cycxlabel_9")
        self.verticalLayout_21.addWidget(self.cycxlabel_9)
        self.ptn_refi_pre = QtWidgets.QLineEdit(parent=self.groupBox_2)
        self.ptn_refi_pre.setMinimumSize(QtCore.QSize(125, 30))
        self.ptn_refi_pre.setMaximumSize(QtCore.QSize(125, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.ptn_refi_pre.setFont(font)
        self.ptn_refi_pre.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhFormattedNumbersOnly)
        self.ptn_refi_pre.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.ptn_refi_pre.setObjectName("ptn_refi_pre")
        self.verticalLayout_21.addWidget(self.ptn_refi_pre)
        self.cycxlabel_10 = QtWidgets.QLabel(parent=self.groupBox_2)
        self.cycxlabel_10.setMinimumSize(QtCore.QSize(125, 20))
        self.cycxlabel_10.setMaximumSize(QtCore.QSize(125, 20))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.cycxlabel_10.setFont(font)
        self.cycxlabel_10.setObjectName("cycxlabel_10")
        self.verticalLayout_21.addWidget(self.cycxlabel_10)
        self.ptn_refi_after = QtWidgets.QLineEdit(parent=self.groupBox_2)
        self.ptn_refi_after.setMinimumSize(QtCore.QSize(125, 30))
        self.ptn_refi_after.setMaximumSize(QtCore.QSize(125, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.ptn_refi_after.setFont(font)
        self.ptn_refi_after.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhFormattedNumbersOnly)
        self.ptn_refi_after.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.ptn_refi_after.setObjectName("ptn_refi_after")
        self.verticalLayout_21.addWidget(self.ptn_refi_after)
        spacerItem5 = QtWidgets.QSpacerItem(118, 50, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Fixed)
        self.verticalLayout_21.addItem(spacerItem5)
        self.chg_ptn_refi = QtWidgets.QPushButton(parent=self.groupBox_2)
        self.chg_ptn_refi.setMinimumSize(QtCore.QSize(125, 50))
        self.chg_ptn_refi.setMaximumSize(QtCore.QSize(125, 50))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.chg_ptn_refi.setFont(font)
        self.chg_ptn_refi.setObjectName("chg_ptn_refi")
        self.verticalLayout_21.addWidget(self.chg_ptn_refi)
        self.gridLayout_12.addLayout(self.verticalLayout_21, 0, 0, 1, 1)
        self.verticalLayout_70.addWidget(self.groupBox_2)
        self.groupBox_7 = QtWidgets.QGroupBox(parent=self.tab_2)
        self.groupBox_7.setMinimumSize(QtCore.QSize(140, 300))
        self.groupBox_7.setMaximumSize(QtCore.QSize(140, 300))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.groupBox_7.setFont(font)
        self.groupBox_7.setObjectName("groupBox_7")
        self.gridLayout_21 = QtWidgets.QGridLayout(self.groupBox_7)
        self.gridLayout_21.setObjectName("gridLayout_21")
        self.verticalLayout_35 = QtWidgets.QVBoxLayout()
        self.verticalLayout_35.setObjectName("verticalLayout_35")
        self.cycxlabel_27 = QtWidgets.QLabel(parent=self.groupBox_7)
        self.cycxlabel_27.setMinimumSize(QtCore.QSize(125, 20))
        self.cycxlabel_27.setMaximumSize(QtCore.QSize(125, 20))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.cycxlabel_27.setFont(font)
        self.cycxlabel_27.setObjectName("cycxlabel_27")
        self.verticalLayout_35.addWidget(self.cycxlabel_27)
        self.ptn_dchgv_pre = QtWidgets.QLineEdit(parent=self.groupBox_7)
        self.ptn_dchgv_pre.setMinimumSize(QtCore.QSize(125, 30))
        self.ptn_dchgv_pre.setMaximumSize(QtCore.QSize(125, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.ptn_dchgv_pre.setFont(font)
        self.ptn_dchgv_pre.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhFormattedNumbersOnly)
        self.ptn_dchgv_pre.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.ptn_dchgv_pre.setObjectName("ptn_dchgv_pre")
        self.verticalLayout_35.addWidget(self.ptn_dchgv_pre)
        self.cycxlabel_28 = QtWidgets.QLabel(parent=self.groupBox_7)
        self.cycxlabel_28.setMinimumSize(QtCore.QSize(125, 20))
        self.cycxlabel_28.setMaximumSize(QtCore.QSize(125, 20))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.cycxlabel_28.setFont(font)
        self.cycxlabel_28.setObjectName("cycxlabel_28")
        self.verticalLayout_35.addWidget(self.cycxlabel_28)
        self.ptn_dchgv_after = QtWidgets.QLineEdit(parent=self.groupBox_7)
        self.ptn_dchgv_after.setMinimumSize(QtCore.QSize(125, 30))
        self.ptn_dchgv_after.setMaximumSize(QtCore.QSize(125, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.ptn_dchgv_after.setFont(font)
        self.ptn_dchgv_after.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhFormattedNumbersOnly)
        self.ptn_dchgv_after.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.ptn_dchgv_after.setObjectName("ptn_dchgv_after")
        self.verticalLayout_35.addWidget(self.ptn_dchgv_after)
        spacerItem6 = QtWidgets.QSpacerItem(118, 50, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Fixed)
        self.verticalLayout_35.addItem(spacerItem6)
        self.chg_ptn_dchgv = QtWidgets.QPushButton(parent=self.groupBox_7)
        self.chg_ptn_dchgv.setMinimumSize(QtCore.QSize(125, 50))
        self.chg_ptn_dchgv.setMaximumSize(QtCore.QSize(125, 50))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.chg_ptn_dchgv.setFont(font)
        self.chg_ptn_dchgv.setObjectName("chg_ptn_dchgv")
        self.verticalLayout_35.addWidget(self.chg_ptn_dchgv)
        self.gridLayout_21.addLayout(self.verticalLayout_35, 0, 0, 1, 1)
        self.verticalLayout_70.addWidget(self.groupBox_7)
        self.horizontalLayout_133.addLayout(self.verticalLayout_70)
        spacerItem7 = QtWidgets.QSpacerItem(18, 608, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.horizontalLayout_133.addItem(spacerItem7)
        self.verticalLayout_71 = QtWidgets.QVBoxLayout()
        self.verticalLayout_71.setObjectName("verticalLayout_71")
        self.groupBox_3 = QtWidgets.QGroupBox(parent=self.tab_2)
        self.groupBox_3.setMinimumSize(QtCore.QSize(140, 300))
        self.groupBox_3.setMaximumSize(QtCore.QSize(140, 300))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.groupBox_3.setFont(font)
        self.groupBox_3.setObjectName("groupBox_3")
        self.gridLayout_13 = QtWidgets.QGridLayout(self.groupBox_3)
        self.gridLayout_13.setObjectName("gridLayout_13")
        self.verticalLayout_23 = QtWidgets.QVBoxLayout()
        self.verticalLayout_23.setObjectName("verticalLayout_23")
        self.cycxlabel_11 = QtWidgets.QLabel(parent=self.groupBox_3)
        self.cycxlabel_11.setMinimumSize(QtCore.QSize(125, 20))
        self.cycxlabel_11.setMaximumSize(QtCore.QSize(125, 20))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.cycxlabel_11.setFont(font)
        self.cycxlabel_11.setObjectName("cycxlabel_11")
        self.verticalLayout_23.addWidget(self.cycxlabel_11)
        self.ptn_endi_pre = QtWidgets.QLineEdit(parent=self.groupBox_3)
        self.ptn_endi_pre.setMinimumSize(QtCore.QSize(125, 30))
        self.ptn_endi_pre.setMaximumSize(QtCore.QSize(125, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.ptn_endi_pre.setFont(font)
        self.ptn_endi_pre.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhFormattedNumbersOnly)
        self.ptn_endi_pre.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.ptn_endi_pre.setObjectName("ptn_endi_pre")
        self.verticalLayout_23.addWidget(self.ptn_endi_pre)
        self.cycxlabel_13 = QtWidgets.QLabel(parent=self.groupBox_3)
        self.cycxlabel_13.setMinimumSize(QtCore.QSize(125, 20))
        self.cycxlabel_13.setMaximumSize(QtCore.QSize(125, 20))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.cycxlabel_13.setFont(font)
        self.cycxlabel_13.setObjectName("cycxlabel_13")
        self.verticalLayout_23.addWidget(self.cycxlabel_13)
        self.ptn_endi_after = QtWidgets.QLineEdit(parent=self.groupBox_3)
        self.ptn_endi_after.setMinimumSize(QtCore.QSize(125, 30))
        self.ptn_endi_after.setMaximumSize(QtCore.QSize(125, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.ptn_endi_after.setFont(font)
        self.ptn_endi_after.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhFormattedNumbersOnly)
        self.ptn_endi_after.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.ptn_endi_after.setObjectName("ptn_endi_after")
        self.verticalLayout_23.addWidget(self.ptn_endi_after)
        spacerItem8 = QtWidgets.QSpacerItem(118, 50, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Fixed)
        self.verticalLayout_23.addItem(spacerItem8)
        self.chg_ptn_endi = QtWidgets.QPushButton(parent=self.groupBox_3)
        self.chg_ptn_endi.setMinimumSize(QtCore.QSize(125, 50))
        self.chg_ptn_endi.setMaximumSize(QtCore.QSize(125, 50))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.chg_ptn_endi.setFont(font)
        self.chg_ptn_endi.setObjectName("chg_ptn_endi")
        self.verticalLayout_23.addWidget(self.chg_ptn_endi)
        self.gridLayout_13.addLayout(self.verticalLayout_23, 0, 0, 1, 1)
        self.verticalLayout_71.addWidget(self.groupBox_3)
        self.groupBox_6 = QtWidgets.QGroupBox(parent=self.tab_2)
        self.groupBox_6.setMinimumSize(QtCore.QSize(140, 300))
        self.groupBox_6.setMaximumSize(QtCore.QSize(140, 300))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.groupBox_6.setFont(font)
        self.groupBox_6.setObjectName("groupBox_6")
        self.gridLayout_18 = QtWidgets.QGridLayout(self.groupBox_6)
        self.gridLayout_18.setObjectName("gridLayout_18")
        self.verticalLayout_32 = QtWidgets.QVBoxLayout()
        self.verticalLayout_32.setObjectName("verticalLayout_32")
        self.cycxlabel_21 = QtWidgets.QLabel(parent=self.groupBox_6)
        self.cycxlabel_21.setMinimumSize(QtCore.QSize(125, 20))
        self.cycxlabel_21.setMaximumSize(QtCore.QSize(125, 20))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.cycxlabel_21.setFont(font)
        self.cycxlabel_21.setObjectName("cycxlabel_21")
        self.verticalLayout_32.addWidget(self.cycxlabel_21)
        self.ptn_endv_pre = QtWidgets.QLineEdit(parent=self.groupBox_6)
        self.ptn_endv_pre.setMinimumSize(QtCore.QSize(125, 30))
        self.ptn_endv_pre.setMaximumSize(QtCore.QSize(125, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.ptn_endv_pre.setFont(font)
        self.ptn_endv_pre.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhFormattedNumbersOnly)
        self.ptn_endv_pre.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.ptn_endv_pre.setObjectName("ptn_endv_pre")
        self.verticalLayout_32.addWidget(self.ptn_endv_pre)
        self.cycxlabel_22 = QtWidgets.QLabel(parent=self.groupBox_6)
        self.cycxlabel_22.setMinimumSize(QtCore.QSize(125, 20))
        self.cycxlabel_22.setMaximumSize(QtCore.QSize(125, 20))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.cycxlabel_22.setFont(font)
        self.cycxlabel_22.setObjectName("cycxlabel_22")
        self.verticalLayout_32.addWidget(self.cycxlabel_22)
        self.ptn_endv_after = QtWidgets.QLineEdit(parent=self.groupBox_6)
        self.ptn_endv_after.setMinimumSize(QtCore.QSize(125, 30))
        self.ptn_endv_after.setMaximumSize(QtCore.QSize(125, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.ptn_endv_after.setFont(font)
        self.ptn_endv_after.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhFormattedNumbersOnly)
        self.ptn_endv_after.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.ptn_endv_after.setObjectName("ptn_endv_after")
        self.verticalLayout_32.addWidget(self.ptn_endv_after)
        spacerItem9 = QtWidgets.QSpacerItem(118, 50, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Fixed)
        self.verticalLayout_32.addItem(spacerItem9)
        self.chg_ptn_endv = QtWidgets.QPushButton(parent=self.groupBox_6)
        self.chg_ptn_endv.setMinimumSize(QtCore.QSize(125, 50))
        self.chg_ptn_endv.setMaximumSize(QtCore.QSize(125, 50))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.chg_ptn_endv.setFont(font)
        self.chg_ptn_endv.setObjectName("chg_ptn_endv")
        self.verticalLayout_32.addWidget(self.chg_ptn_endv)
        self.gridLayout_18.addLayout(self.verticalLayout_32, 0, 0, 1, 1)
        self.verticalLayout_71.addWidget(self.groupBox_6)
        self.horizontalLayout_133.addLayout(self.verticalLayout_71)
        spacerItem10 = QtWidgets.QSpacerItem(17, 608, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.horizontalLayout_133.addItem(spacerItem10)
        self.groupBox_4 = QtWidgets.QGroupBox(parent=self.tab_2)
        self.groupBox_4.setMinimumSize(QtCore.QSize(140, 400))
        self.groupBox_4.setMaximumSize(QtCore.QSize(140, 400))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.groupBox_4.setFont(font)
        self.groupBox_4.setObjectName("groupBox_4")
        self.gridLayout_14 = QtWidgets.QGridLayout(self.groupBox_4)
        self.gridLayout_14.setObjectName("gridLayout_14")
        self.verticalLayout_30 = QtWidgets.QVBoxLayout()
        self.verticalLayout_30.setObjectName("verticalLayout_30")
        self.cycxlabel_16 = QtWidgets.QLabel(parent=self.groupBox_4)
        self.cycxlabel_16.setMinimumSize(QtCore.QSize(125, 20))
        self.cycxlabel_16.setMaximumSize(QtCore.QSize(125, 20))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.cycxlabel_16.setFont(font)
        self.cycxlabel_16.setObjectName("cycxlabel_16")
        self.verticalLayout_30.addWidget(self.cycxlabel_16)
        self.ptn_step_pre = QtWidgets.QLineEdit(parent=self.groupBox_4)
        self.ptn_step_pre.setMinimumSize(QtCore.QSize(125, 30))
        self.ptn_step_pre.setMaximumSize(QtCore.QSize(125, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.ptn_step_pre.setFont(font)
        self.ptn_step_pre.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhFormattedNumbersOnly)
        self.ptn_step_pre.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.ptn_step_pre.setObjectName("ptn_step_pre")
        self.verticalLayout_30.addWidget(self.ptn_step_pre)
        self.cycxlabel_14 = QtWidgets.QLabel(parent=self.groupBox_4)
        self.cycxlabel_14.setMinimumSize(QtCore.QSize(125, 20))
        self.cycxlabel_14.setMaximumSize(QtCore.QSize(125, 20))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.cycxlabel_14.setFont(font)
        self.cycxlabel_14.setObjectName("cycxlabel_14")
        self.verticalLayout_30.addWidget(self.cycxlabel_14)
        self.ptn_step_after = QtWidgets.QLineEdit(parent=self.groupBox_4)
        self.ptn_step_after.setMinimumSize(QtCore.QSize(125, 30))
        self.ptn_step_after.setMaximumSize(QtCore.QSize(125, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.ptn_step_after.setFont(font)
        self.ptn_step_after.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhFormattedNumbersOnly)
        self.ptn_step_after.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.ptn_step_after.setObjectName("ptn_step_after")
        self.verticalLayout_30.addWidget(self.ptn_step_after)
        spacerItem11 = QtWidgets.QSpacerItem(118, 80, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Fixed)
        self.verticalLayout_30.addItem(spacerItem11)
        self.chg_ptn_step = QtWidgets.QPushButton(parent=self.groupBox_4)
        self.chg_ptn_step.setMinimumSize(QtCore.QSize(125, 50))
        self.chg_ptn_step.setMaximumSize(QtCore.QSize(125, 50))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.chg_ptn_step.setFont(font)
        self.chg_ptn_step.setObjectName("chg_ptn_step")
        self.verticalLayout_30.addWidget(self.chg_ptn_step)
        self.gridLayout_14.addLayout(self.verticalLayout_30, 0, 0, 1, 1)
        self.horizontalLayout_133.addWidget(self.groupBox_4)
        self.verticalLayout_9.addLayout(self.horizontalLayout_133)
        self.horizontalLayout_134.addLayout(self.verticalLayout_9)
        self.ptn_list = QtWidgets.QTableWidget(parent=self.tab_2)
        self.ptn_list.setMinimumSize(QtCore.QSize(1200, 830))
        self.ptn_list.setMaximumSize(QtCore.QSize(1200, 830))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.ptn_list.setFont(font)
        self.ptn_list.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.SizeAdjustPolicy.AdjustToContents)
        self.ptn_list.setAlternatingRowColors(True)
        self.ptn_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.ptn_list.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.ptn_list.setObjectName("ptn_list")
        self.ptn_list.setColumnCount(4)
        self.ptn_list.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setFamily("malgun gothic")
        font.setPointSize(8)
        item.setFont(font)
        self.ptn_list.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setFamily("malgun gothic")
        font.setPointSize(8)
        item.setFont(font)
        self.ptn_list.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setFamily("malgun gothic")
        font.setPointSize(8)
        item.setFont(font)
        self.ptn_list.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setFamily("malgun gothic")
        font.setPointSize(8)
        item.setFont(font)
        self.ptn_list.setHorizontalHeaderItem(3, item)
        self.ptn_list.horizontalHeader().setVisible(False)
        self.ptn_list.horizontalHeader().setDefaultSectionSize(198)
        self.ptn_list.horizontalHeader().setMinimumSectionSize(50)
        self.ptn_list.verticalHeader().setVisible(False)
        self.horizontalLayout_134.addWidget(self.ptn_list)
        self.horizontalLayout_135.addLayout(self.horizontalLayout_134)
        self.tabWidget.addTab(self.tab_2, "")
        self.SetTab = QtWidgets.QWidget()
        self.SetTab.setObjectName("SetTab")
        self.horizontalLayout_190 = QtWidgets.QHBoxLayout(self.SetTab)
        self.horizontalLayout_190.setObjectName("horizontalLayout_190")
        self.horizontalLayout_155 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_155.setObjectName("horizontalLayout_155")
        self.verticalLayout_20 = QtWidgets.QVBoxLayout()
        self.verticalLayout_20.setObjectName("verticalLayout_20")
        self.horizontalLayout_124 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_124.setObjectName("horizontalLayout_124")
        self.verticalLayout_19 = QtWidgets.QVBoxLayout()
        self.verticalLayout_19.setObjectName("verticalLayout_19")
        self.Capacitynum = QtWidgets.QGroupBox(parent=self.SetTab)
        self.Capacitynum.setMinimumSize(QtCore.QSize(320, 20))
        self.Capacitynum.setMaximumSize(QtCore.QSize(320, 100))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.Capacitynum.setFont(font)
        self.Capacitynum.setObjectName("Capacitynum")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.Capacitynum)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.SetMincapacity = QtWidgets.QLineEdit(parent=self.Capacitynum)
        self.SetMincapacity.setMinimumSize(QtCore.QSize(300, 20))
        self.SetMincapacity.setMaximumSize(QtCore.QSize(300, 20))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.SetMincapacity.setFont(font)
        self.SetMincapacity.setObjectName("SetMincapacity")
        self.verticalLayout_7.addWidget(self.SetMincapacity)
        self.verticalLayout_19.addWidget(self.Capacitynum)
        self.Capacitynum_2 = QtWidgets.QGroupBox(parent=self.SetTab)
        self.Capacitynum_2.setMinimumSize(QtCore.QSize(320, 20))
        self.Capacitynum_2.setMaximumSize(QtCore.QSize(320, 100))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.Capacitynum_2.setFont(font)
        self.Capacitynum_2.setObjectName("Capacitynum_2")
        self.verticalLayout_40 = QtWidgets.QVBoxLayout(self.Capacitynum_2)
        self.verticalLayout_40.setObjectName("verticalLayout_40")
        self.SetMaxCycle = QtWidgets.QLineEdit(parent=self.Capacitynum_2)
        self.SetMaxCycle.setMinimumSize(QtCore.QSize(300, 20))
        self.SetMaxCycle.setMaximumSize(QtCore.QSize(300, 20))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.SetMaxCycle.setFont(font)
        self.SetMaxCycle.setObjectName("SetMaxCycle")
        self.verticalLayout_40.addWidget(self.SetMaxCycle)
        self.verticalLayout_19.addWidget(self.Capacitynum_2)
        self.gCyclesetting = QtWidgets.QGroupBox(parent=self.SetTab)
        self.gCyclesetting.setMinimumSize(QtCore.QSize(320, 256))
        self.gCyclesetting.setMaximumSize(QtCore.QSize(320, 256))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.gCyclesetting.setFont(font)
        self.gCyclesetting.setObjectName("gCyclesetting")
        self.horizontalLayout_72 = QtWidgets.QHBoxLayout(self.gCyclesetting)
        self.horizontalLayout_72.setObjectName("horizontalLayout_72")
        self.verticalLayout_12 = QtWidgets.QVBoxLayout()
        self.verticalLayout_12.setObjectName("verticalLayout_12")
        self.realcyc = QtWidgets.QRadioButton(parent=self.gCyclesetting)
        self.realcyc.setMinimumSize(QtCore.QSize(87, 20))
        self.realcyc.setMaximumSize(QtCore.QSize(87, 20))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.realcyc.setFont(font)
        self.realcyc.setObjectName("realcyc")
        self.verticalLayout_12.addWidget(self.realcyc)
        self.resetcycle = QtWidgets.QRadioButton(parent=self.gCyclesetting)
        self.resetcycle.setMinimumSize(QtCore.QSize(83, 20))
        self.resetcycle.setMaximumSize(QtCore.QSize(83, 20))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.resetcycle.setFont(font)
        self.resetcycle.setChecked(True)
        self.resetcycle.setObjectName("resetcycle")
        self.verticalLayout_12.addWidget(self.resetcycle)
        self.line_4 = QtWidgets.QFrame(parent=self.gCyclesetting)
        self.line_4.setMinimumSize(QtCore.QSize(298, 3))
        self.line_4.setMaximumSize(QtCore.QSize(298, 3))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.line_4.setFont(font)
        self.line_4.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        self.line_4.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.line_4.setObjectName("line_4")
        self.verticalLayout_12.addWidget(self.line_4)
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.cycxlabel_4 = QtWidgets.QLabel(parent=self.gCyclesetting)
        self.cycxlabel_4.setMinimumSize(QtCore.QSize(190, 20))
        self.cycxlabel_4.setMaximumSize(QtCore.QSize(190, 20))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.cycxlabel_4.setFont(font)
        self.cycxlabel_4.setObjectName("cycxlabel_4")
        self.horizontalLayout_9.addWidget(self.cycxlabel_4)
        self.setcyclexscale = QtWidgets.QLineEdit(parent=self.gCyclesetting)
        self.setcyclexscale.setMinimumSize(QtCore.QSize(100, 20))
        self.setcyclexscale.setMaximumSize(QtCore.QSize(100, 20))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.setcyclexscale.setFont(font)
        self.setcyclexscale.setObjectName("setcyclexscale")
        self.horizontalLayout_9.addWidget(self.setcyclexscale)
        self.verticalLayout_12.addLayout(self.horizontalLayout_9)
        self.allcycle = QtWidgets.QRadioButton(parent=self.gCyclesetting)
        self.allcycle.setMinimumSize(QtCore.QSize(297, 20))
        self.allcycle.setMaximumSize(QtCore.QSize(297, 20))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.allcycle.setFont(font)
        self.allcycle.setObjectName("allcycle")
        self.verticalLayout_12.addWidget(self.allcycle)
        self.horizontalLayout_70 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_70.setObjectName("horizontalLayout_70")
        self.recentcycle = QtWidgets.QRadioButton(parent=self.gCyclesetting)
        self.recentcycle.setMinimumSize(QtCore.QSize(190, 20))
        self.recentcycle.setMaximumSize(QtCore.QSize(190, 20))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.recentcycle.setFont(font)
        self.recentcycle.setChecked(True)
        self.recentcycle.setObjectName("recentcycle")
        self.horizontalLayout_70.addWidget(self.recentcycle)
        self.recentcycleno = QtWidgets.QLineEdit(parent=self.gCyclesetting)
        self.recentcycleno.setMinimumSize(QtCore.QSize(100, 20))
        self.recentcycleno.setMaximumSize(QtCore.QSize(100, 20))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.recentcycleno.setFont(font)
        self.recentcycleno.setObjectName("recentcycleno")
        self.horizontalLayout_70.addWidget(self.recentcycleno)
        self.verticalLayout_12.addLayout(self.horizontalLayout_70)
        self.horizontalLayout_71 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_71.setObjectName("horizontalLayout_71")
        self.manualcycle = QtWidgets.QRadioButton(parent=self.gCyclesetting)
        self.manualcycle.setMinimumSize(QtCore.QSize(190, 30))
        self.manualcycle.setMaximumSize(QtCore.QSize(190, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.manualcycle.setFont(font)
        self.manualcycle.setChecked(False)
        self.manualcycle.setObjectName("manualcycle")
        self.horizontalLayout_71.addWidget(self.manualcycle)
        self.manualcycleno = QtWidgets.QLineEdit(parent=self.gCyclesetting)
        self.manualcycleno.setMinimumSize(QtCore.QSize(100, 20))
        self.manualcycleno.setMaximumSize(QtCore.QSize(100, 20))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.manualcycleno.setFont(font)
        self.manualcycleno.setObjectName("manualcycleno")
        self.horizontalLayout_71.addWidget(self.manualcycleno)
        self.verticalLayout_12.addLayout(self.horizontalLayout_71)
        self.horizontalLayout_72.addLayout(self.verticalLayout_12)
        self.verticalLayout_19.addWidget(self.gCyclesetting)
        self.horizontalLayout_124.addLayout(self.verticalLayout_19)
        self.groupBox = QtWidgets.QGroupBox(parent=self.SetTab)
        self.groupBox.setMinimumSize(QtCore.QSize(320, 364))
        self.groupBox.setMaximumSize(QtCore.QSize(320, 364))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.groupBox.setFont(font)
        self.groupBox.setObjectName("groupBox")
        self.horizontalLayout_73 = QtWidgets.QHBoxLayout(self.groupBox)
        self.horizontalLayout_73.setObjectName("horizontalLayout_73")
        self.verticalLayout_27 = QtWidgets.QVBoxLayout()
        self.verticalLayout_27.setObjectName("verticalLayout_27")
        self.label_3 = QtWidgets.QLabel(parent=self.groupBox)
        self.label_3.setMinimumSize(QtCore.QSize(298, 20))
        self.label_3.setMaximumSize(QtCore.QSize(298, 20))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.verticalLayout_27.addWidget(self.label_3)
        self.socmaxcapacity = QtWidgets.QLineEdit(parent=self.groupBox)
        self.socmaxcapacity.setMinimumSize(QtCore.QSize(298, 20))
        self.socmaxcapacity.setMaximumSize(QtCore.QSize(298, 20))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.socmaxcapacity.setFont(font)
        self.socmaxcapacity.setText("")
        self.socmaxcapacity.setObjectName("socmaxcapacity")
        self.verticalLayout_27.addWidget(self.socmaxcapacity)
        self.label_11 = QtWidgets.QLabel(parent=self.groupBox)
        self.label_11.setMinimumSize(QtCore.QSize(298, 20))
        self.label_11.setMaximumSize(QtCore.QSize(298, 20))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.label_11.setFont(font)
        self.label_11.setObjectName("label_11")
        self.verticalLayout_27.addWidget(self.label_11)
        self.setoffvoltage = QtWidgets.QLineEdit(parent=self.groupBox)
        self.setoffvoltage.setMinimumSize(QtCore.QSize(298, 20))
        self.setoffvoltage.setMaximumSize(QtCore.QSize(298, 20))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.setoffvoltage.setFont(font)
        self.setoffvoltage.setText("")
        self.setoffvoltage.setObjectName("setoffvoltage")
        self.verticalLayout_27.addWidget(self.setoffvoltage)
        spacerItem12 = QtWidgets.QSpacerItem(298, 112, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Expanding)
        self.verticalLayout_27.addItem(spacerItem12)
        self.horizontalLayout_81 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_81.setObjectName("horizontalLayout_81")
        self.label_12 = QtWidgets.QLabel(parent=self.groupBox)
        self.label_12.setMinimumSize(QtCore.QSize(100, 20))
        self.label_12.setMaximumSize(QtCore.QSize(100, 20))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.label_12.setFont(font)
        self.label_12.setObjectName("label_12")
        self.horizontalLayout_81.addWidget(self.label_12)
        self.label_14 = QtWidgets.QLabel(parent=self.groupBox)
        self.label_14.setMinimumSize(QtCore.QSize(100, 20))
        self.label_14.setMaximumSize(QtCore.QSize(100, 20))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.label_14.setFont(font)
        self.label_14.setObjectName("label_14")
        self.horizontalLayout_81.addWidget(self.label_14)
        self.verticalLayout_27.addLayout(self.horizontalLayout_81)
        self.horizontalLayout_82 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_82.setObjectName("horizontalLayout_82")
        self.socerrormax = QtWidgets.QLineEdit(parent=self.groupBox)
        self.socerrormax.setMinimumSize(QtCore.QSize(100, 20))
        self.socerrormax.setMaximumSize(QtCore.QSize(100, 20))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.socerrormax.setFont(font)
        self.socerrormax.setText("")
        self.socerrormax.setObjectName("socerrormax")
        self.horizontalLayout_82.addWidget(self.socerrormax)
        self.socerroravg = QtWidgets.QLineEdit(parent=self.groupBox)
        self.socerroravg.setMinimumSize(QtCore.QSize(100, 20))
        self.socerroravg.setMaximumSize(QtCore.QSize(100, 20))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.socerroravg.setFont(font)
        self.socerroravg.setText("")
        self.socerroravg.setObjectName("socerroravg")
        self.horizontalLayout_82.addWidget(self.socerroravg)
        self.verticalLayout_27.addLayout(self.horizontalLayout_82)
        self.horizontalLayout_83 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_83.setObjectName("horizontalLayout_83")
        self.label_13 = QtWidgets.QLabel(parent=self.groupBox)
        self.label_13.setMinimumSize(QtCore.QSize(100, 20))
        self.label_13.setMaximumSize(QtCore.QSize(100, 20))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.label_13.setFont(font)
        self.label_13.setObjectName("label_13")
        self.horizontalLayout_83.addWidget(self.label_13)
        self.label_15 = QtWidgets.QLabel(parent=self.groupBox)
        self.label_15.setMinimumSize(QtCore.QSize(100, 20))
        self.label_15.setMaximumSize(QtCore.QSize(100, 20))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.label_15.setFont(font)
        self.label_15.setObjectName("label_15")
        self.horizontalLayout_83.addWidget(self.label_15)
        self.verticalLayout_27.addLayout(self.horizontalLayout_83)
        self.horizontalLayout_84 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_84.setObjectName("horizontalLayout_84")
        self.ectsocerrormax = QtWidgets.QLineEdit(parent=self.groupBox)
        self.ectsocerrormax.setMinimumSize(QtCore.QSize(100, 20))
        self.ectsocerrormax.setMaximumSize(QtCore.QSize(100, 20))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.ectsocerrormax.setFont(font)
        self.ectsocerrormax.setText("")
        self.ectsocerrormax.setObjectName("ectsocerrormax")
        self.horizontalLayout_84.addWidget(self.ectsocerrormax)
        self.ectsocerroravg = QtWidgets.QLineEdit(parent=self.groupBox)
        self.ectsocerroravg.setMinimumSize(QtCore.QSize(100, 20))
        self.ectsocerroravg.setMaximumSize(QtCore.QSize(100, 20))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.ectsocerroravg.setFont(font)
        self.ectsocerroravg.setText("")
        self.ectsocerroravg.setObjectName("ectsocerroravg")
        self.horizontalLayout_84.addWidget(self.ectsocerroravg)
        self.verticalLayout_27.addLayout(self.horizontalLayout_84)
        self.horizontalLayout_73.addLayout(self.verticalLayout_27)
        self.horizontalLayout_124.addWidget(self.groupBox)
        self.verticalLayout_20.addLayout(self.horizontalLayout_124)
        self.verticalLayout_13 = QtWidgets.QVBoxLayout()
        self.verticalLayout_13.setObjectName("verticalLayout_13")
        self.horizontalLayout_125 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_125.setObjectName("horizontalLayout_125")
        self.label_4 = QtWidgets.QLabel(parent=self.SetTab)
        self.label_4.setMinimumSize(QtCore.QSize(320, 70))
        self.label_4.setMaximumSize(QtCore.QSize(320, 70))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.label_4.setFont(font)
        self.label_4.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_125.addWidget(self.label_4)
        self.label_7 = QtWidgets.QLabel(parent=self.SetTab)
        self.label_7.setMinimumSize(QtCore.QSize(320, 70))
        self.label_7.setMaximumSize(QtCore.QSize(320, 70))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.label_7.setFont(font)
        self.label_7.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_7.setObjectName("label_7")
        self.horizontalLayout_125.addWidget(self.label_7)
        self.verticalLayout_13.addLayout(self.horizontalLayout_125)
        self.horizontalLayout_126 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_126.setObjectName("horizontalLayout_126")
        self.SETTabReset = QtWidgets.QPushButton(parent=self.SetTab)
        self.SETTabReset.setMinimumSize(QtCore.QSize(320, 70))
        self.SETTabReset.setMaximumSize(QtCore.QSize(320, 70))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(True)
        font.setItalic(False)
        font.setUnderline(True)
        font.setWeight(75)
        self.SETTabReset.setFont(font)
        self.SETTabReset.setAutoDefault(True)
        self.SETTabReset.setFlat(False)
        self.SETTabReset.setObjectName("SETTabReset")
        self.horizontalLayout_126.addWidget(self.SETTabReset)
        self.ECTSOC = QtWidgets.QPushButton(parent=self.SetTab)
        self.ECTSOC.setMinimumSize(QtCore.QSize(320, 70))
        self.ECTSOC.setMaximumSize(QtCore.QSize(320, 70))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.ECTSOC.setFont(font)
        self.ECTSOC.setObjectName("ECTSOC")
        self.horizontalLayout_126.addWidget(self.ECTSOC)
        self.verticalLayout_13.addLayout(self.horizontalLayout_126)
        self.horizontalLayout_127 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_127.setObjectName("horizontalLayout_127")
        self.SetlogConfirm = QtWidgets.QPushButton(parent=self.SetTab)
        self.SetlogConfirm.setMinimumSize(QtCore.QSize(320, 70))
        self.SetlogConfirm.setMaximumSize(QtCore.QSize(320, 70))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.SetlogConfirm.setFont(font)
        self.SetlogConfirm.setObjectName("SetlogConfirm")
        self.horizontalLayout_127.addWidget(self.SetlogConfirm)
        self.ECTShort = QtWidgets.QPushButton(parent=self.SetTab)
        self.ECTShort.setMinimumSize(QtCore.QSize(320, 70))
        self.ECTShort.setMaximumSize(QtCore.QSize(320, 70))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.ECTShort.setFont(font)
        self.ECTShort.setObjectName("ECTShort")
        self.horizontalLayout_127.addWidget(self.ECTShort)
        self.verticalLayout_13.addLayout(self.horizontalLayout_127)
        self.horizontalLayout_131 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_131.setObjectName("horizontalLayout_131")
        self.SetConfirm = QtWidgets.QPushButton(parent=self.SetTab)
        self.SetConfirm.setMinimumSize(QtCore.QSize(320, 70))
        self.SetConfirm.setMaximumSize(QtCore.QSize(320, 70))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.SetConfirm.setFont(font)
        self.SetConfirm.setObjectName("SetConfirm")
        self.horizontalLayout_131.addWidget(self.SetConfirm)
        self.ECTSetProfile = QtWidgets.QPushButton(parent=self.SetTab)
        self.ECTSetProfile.setMinimumSize(QtCore.QSize(320, 70))
        self.ECTSetProfile.setMaximumSize(QtCore.QSize(320, 70))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.ECTSetProfile.setFont(font)
        self.ECTSetProfile.setObjectName("ECTSetProfile")
        self.horizontalLayout_131.addWidget(self.ECTSetProfile)
        self.verticalLayout_13.addLayout(self.horizontalLayout_131)
        self.horizontalLayout_132 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_132.setObjectName("horizontalLayout_132")
        self.SetCycle = QtWidgets.QPushButton(parent=self.SetTab)
        self.SetCycle.setMinimumSize(QtCore.QSize(320, 70))
        self.SetCycle.setMaximumSize(QtCore.QSize(320, 70))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.SetCycle.setFont(font)
        self.SetCycle.setObjectName("SetCycle")
        self.horizontalLayout_132.addWidget(self.SetCycle)
        self.ECTSetCycle = QtWidgets.QPushButton(parent=self.SetTab)
        self.ECTSetCycle.setMinimumSize(QtCore.QSize(320, 70))
        self.ECTSetCycle.setMaximumSize(QtCore.QSize(320, 70))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.ECTSetCycle.setFont(font)
        self.ECTSetCycle.setObjectName("ECTSetCycle")
        self.horizontalLayout_132.addWidget(self.ECTSetCycle)
        self.verticalLayout_13.addLayout(self.horizontalLayout_132)
        self.horizontalLayout_136 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_136.setObjectName("horizontalLayout_136")
        self.ECTSetlog2 = QtWidgets.QPushButton(parent=self.SetTab)
        self.ECTSetlog2.setMinimumSize(QtCore.QSize(320, 70))
        self.ECTSetlog2.setMaximumSize(QtCore.QSize(320, 70))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.ECTSetlog2.setFont(font)
        self.ECTSetlog2.setObjectName("ECTSetlog2")
        self.horizontalLayout_136.addWidget(self.ECTSetlog2)
        self.ECTSetlog = QtWidgets.QPushButton(parent=self.SetTab)
        self.ECTSetlog.setMinimumSize(QtCore.QSize(320, 70))
        self.ECTSetlog.setMaximumSize(QtCore.QSize(320, 70))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.ECTSetlog.setFont(font)
        self.ECTSetlog.setObjectName("ECTSetlog")
        self.horizontalLayout_136.addWidget(self.ECTSetlog)
        self.verticalLayout_13.addLayout(self.horizontalLayout_136)
        self.verticalLayout_20.addLayout(self.verticalLayout_13)
        self.horizontalLayout_155.addLayout(self.verticalLayout_20)
        self.set_tab = QtWidgets.QTabWidget(parent=self.SetTab)
        self.set_tab.setMinimumSize(QtCore.QSize(1200, 830))
        self.set_tab.setMaximumSize(QtCore.QSize(1200, 830))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.set_tab.setFont(font)
        self.set_tab.setObjectName("set_tab")
        self.horizontalLayout_155.addWidget(self.set_tab)
        self.horizontalLayout_190.addLayout(self.horizontalLayout_155)
        self.horizontalLayout_130 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_130.setObjectName("horizontalLayout_130")
        self.horizontalLayout_190.addLayout(self.horizontalLayout_130)
        self.tabWidget.addTab(self.SetTab, "")
        self.dvdq = QtWidgets.QWidget()
        self.dvdq.setObjectName("dvdq")
        self.horizontalLayout_154 = QtWidgets.QHBoxLayout(self.dvdq)
        self.horizontalLayout_154.setObjectName("horizontalLayout_154")
        self.horizontalLayout_153 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_153.setObjectName("horizontalLayout_153")
        self.verticalLayout_22 = QtWidgets.QVBoxLayout()
        self.verticalLayout_22.setObjectName("verticalLayout_22")
        self.horizontalLayout_137 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_137.setObjectName("horizontalLayout_137")
        self.cycxlabel_29 = QtWidgets.QLabel(parent=self.dvdq)
        self.cycxlabel_29.setMinimumSize(QtCore.QSize(150, 34))
        self.cycxlabel_29.setMaximumSize(QtCore.QSize(150, 34))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.cycxlabel_29.setFont(font)
        self.cycxlabel_29.setObjectName("cycxlabel_29")
        self.horizontalLayout_137.addWidget(self.cycxlabel_29)
        self.ca_mat_dvdq_path = QtWidgets.QLineEdit(parent=self.dvdq)
        self.ca_mat_dvdq_path.setMinimumSize(QtCore.QSize(498, 34))
        self.ca_mat_dvdq_path.setMaximumSize(QtCore.QSize(498, 34))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.ca_mat_dvdq_path.setFont(font)
        self.ca_mat_dvdq_path.setText("")
        self.ca_mat_dvdq_path.setObjectName("ca_mat_dvdq_path")
        self.horizontalLayout_137.addWidget(self.ca_mat_dvdq_path)
        self.verticalLayout_22.addLayout(self.horizontalLayout_137)
        self.horizontalLayout_138 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_138.setObjectName("horizontalLayout_138")
        self.cycxlabel_8 = QtWidgets.QLabel(parent=self.dvdq)
        self.cycxlabel_8.setMinimumSize(QtCore.QSize(150, 33))
        self.cycxlabel_8.setMaximumSize(QtCore.QSize(150, 33))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.cycxlabel_8.setFont(font)
        self.cycxlabel_8.setObjectName("cycxlabel_8")
        self.horizontalLayout_138.addWidget(self.cycxlabel_8)
        self.an_mat_dvdq_path = QtWidgets.QLineEdit(parent=self.dvdq)
        self.an_mat_dvdq_path.setMinimumSize(QtCore.QSize(498, 33))
        self.an_mat_dvdq_path.setMaximumSize(QtCore.QSize(498, 33))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.an_mat_dvdq_path.setFont(font)
        self.an_mat_dvdq_path.setText("")
        self.an_mat_dvdq_path.setObjectName("an_mat_dvdq_path")
        self.horizontalLayout_138.addWidget(self.an_mat_dvdq_path)
        self.verticalLayout_22.addLayout(self.horizontalLayout_138)
        self.horizontalLayout_139 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_139.setObjectName("horizontalLayout_139")
        self.cycxlabel_12 = QtWidgets.QLabel(parent=self.dvdq)
        self.cycxlabel_12.setMinimumSize(QtCore.QSize(150, 34))
        self.cycxlabel_12.setMaximumSize(QtCore.QSize(150, 34))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.cycxlabel_12.setFont(font)
        self.cycxlabel_12.setObjectName("cycxlabel_12")
        self.horizontalLayout_139.addWidget(self.cycxlabel_12)
        self.pro_dvdq_path = QtWidgets.QLineEdit(parent=self.dvdq)
        self.pro_dvdq_path.setMinimumSize(QtCore.QSize(498, 34))
        self.pro_dvdq_path.setMaximumSize(QtCore.QSize(498, 34))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.pro_dvdq_path.setFont(font)
        self.pro_dvdq_path.setText("")
        self.pro_dvdq_path.setObjectName("pro_dvdq_path")
        self.horizontalLayout_139.addWidget(self.pro_dvdq_path)
        self.verticalLayout_22.addLayout(self.horizontalLayout_139)
        self.line_10 = QtWidgets.QFrame(parent=self.dvdq)
        self.line_10.setMinimumSize(QtCore.QSize(656, 3))
        self.line_10.setMaximumSize(QtCore.QSize(656, 3))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.line_10.setFont(font)
        self.line_10.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        self.line_10.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.line_10.setObjectName("line_10")
        self.verticalLayout_22.addWidget(self.line_10)
        self.horizontalLayout_140 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_140.setObjectName("horizontalLayout_140")
        self.cycxlabel_62 = QtWidgets.QLabel(parent=self.dvdq)
        self.cycxlabel_62.setMinimumSize(QtCore.QSize(338, 33))
        self.cycxlabel_62.setMaximumSize(QtCore.QSize(338, 33))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.cycxlabel_62.setFont(font)
        self.cycxlabel_62.setObjectName("cycxlabel_62")
        self.horizontalLayout_140.addWidget(self.cycxlabel_62)
        self.dvdq_start_soc = QtWidgets.QLineEdit(parent=self.dvdq)
        self.dvdq_start_soc.setMinimumSize(QtCore.QSize(310, 33))
        self.dvdq_start_soc.setMaximumSize(QtCore.QSize(310, 33))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.dvdq_start_soc.setFont(font)
        self.dvdq_start_soc.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhFormattedNumbersOnly)
        self.dvdq_start_soc.setObjectName("dvdq_start_soc")
        self.horizontalLayout_140.addWidget(self.dvdq_start_soc)
        self.verticalLayout_22.addLayout(self.horizontalLayout_140)
        self.horizontalLayout_141 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_141.setObjectName("horizontalLayout_141")
        self.cycxlabel_61 = QtWidgets.QLabel(parent=self.dvdq)
        self.cycxlabel_61.setMinimumSize(QtCore.QSize(338, 34))
        self.cycxlabel_61.setMaximumSize(QtCore.QSize(338, 34))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.cycxlabel_61.setFont(font)
        self.cycxlabel_61.setObjectName("cycxlabel_61")
        self.horizontalLayout_141.addWidget(self.cycxlabel_61)
        self.dvdq_end_soc = QtWidgets.QLineEdit(parent=self.dvdq)
        self.dvdq_end_soc.setMinimumSize(QtCore.QSize(310, 34))
        self.dvdq_end_soc.setMaximumSize(QtCore.QSize(310, 34))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.dvdq_end_soc.setFont(font)
        self.dvdq_end_soc.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhFormattedNumbersOnly)
        self.dvdq_end_soc.setObjectName("dvdq_end_soc")
        self.horizontalLayout_141.addWidget(self.dvdq_end_soc)
        self.verticalLayout_22.addLayout(self.horizontalLayout_141)
        self.horizontalLayout_142 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_142.setObjectName("horizontalLayout_142")
        self.cycxlabel_60 = QtWidgets.QLabel(parent=self.dvdq)
        self.cycxlabel_60.setMinimumSize(QtCore.QSize(338, 33))
        self.cycxlabel_60.setMaximumSize(QtCore.QSize(338, 33))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.cycxlabel_60.setFont(font)
        self.cycxlabel_60.setObjectName("cycxlabel_60")
        self.horizontalLayout_142.addWidget(self.cycxlabel_60)
        self.dvdq_full_smoothing_no = QtWidgets.QLineEdit(parent=self.dvdq)
        self.dvdq_full_smoothing_no.setMinimumSize(QtCore.QSize(310, 33))
        self.dvdq_full_smoothing_no.setMaximumSize(QtCore.QSize(310, 33))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.dvdq_full_smoothing_no.setFont(font)
        self.dvdq_full_smoothing_no.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhFormattedNumbersOnly)
        self.dvdq_full_smoothing_no.setObjectName("dvdq_full_smoothing_no")
        self.horizontalLayout_142.addWidget(self.dvdq_full_smoothing_no)
        self.verticalLayout_22.addLayout(self.horizontalLayout_142)
        self.line_11 = QtWidgets.QFrame(parent=self.dvdq)
        self.line_11.setMinimumSize(QtCore.QSize(656, 3))
        self.line_11.setMaximumSize(QtCore.QSize(656, 3))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.line_11.setFont(font)
        self.line_11.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        self.line_11.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.line_11.setObjectName("line_11")
        self.verticalLayout_22.addWidget(self.line_11)
        self.horizontalLayout_143 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_143.setObjectName("horizontalLayout_143")
        self.cycxlabel_32 = QtWidgets.QLabel(parent=self.dvdq)
        self.cycxlabel_32.setMinimumSize(QtCore.QSize(338, 34))
        self.cycxlabel_32.setMaximumSize(QtCore.QSize(338, 34))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.cycxlabel_32.setFont(font)
        self.cycxlabel_32.setObjectName("cycxlabel_32")
        self.horizontalLayout_143.addWidget(self.cycxlabel_32)
        self.full_cell_max_cap_txt = QtWidgets.QLineEdit(parent=self.dvdq)
        self.full_cell_max_cap_txt.setMinimumSize(QtCore.QSize(310, 34))
        self.full_cell_max_cap_txt.setMaximumSize(QtCore.QSize(310, 34))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.full_cell_max_cap_txt.setFont(font)
        self.full_cell_max_cap_txt.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhFormattedNumbersOnly)
        self.full_cell_max_cap_txt.setText("")
        self.full_cell_max_cap_txt.setObjectName("full_cell_max_cap_txt")
        self.horizontalLayout_143.addWidget(self.full_cell_max_cap_txt)
        self.verticalLayout_22.addLayout(self.horizontalLayout_143)
        self.horizontalLayout_144 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_144.setObjectName("horizontalLayout_144")
        self.cycxlabel_30 = QtWidgets.QLabel(parent=self.dvdq)
        self.cycxlabel_30.setMinimumSize(QtCore.QSize(338, 33))
        self.cycxlabel_30.setMaximumSize(QtCore.QSize(338, 33))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.cycxlabel_30.setFont(font)
        self.cycxlabel_30.setObjectName("cycxlabel_30")
        self.horizontalLayout_144.addWidget(self.cycxlabel_30)
        self.ca_max_cap_txt = QtWidgets.QLineEdit(parent=self.dvdq)
        self.ca_max_cap_txt.setMinimumSize(QtCore.QSize(310, 33))
        self.ca_max_cap_txt.setMaximumSize(QtCore.QSize(310, 33))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.ca_max_cap_txt.setFont(font)
        self.ca_max_cap_txt.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhFormattedNumbersOnly)
        self.ca_max_cap_txt.setText("")
        self.ca_max_cap_txt.setObjectName("ca_max_cap_txt")
        self.horizontalLayout_144.addWidget(self.ca_max_cap_txt)
        self.verticalLayout_22.addLayout(self.horizontalLayout_144)
        self.horizontalLayout_145 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_145.setObjectName("horizontalLayout_145")
        self.cycxlabel_31 = QtWidgets.QLabel(parent=self.dvdq)
        self.cycxlabel_31.setMinimumSize(QtCore.QSize(338, 34))
        self.cycxlabel_31.setMaximumSize(QtCore.QSize(338, 34))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.cycxlabel_31.setFont(font)
        self.cycxlabel_31.setObjectName("cycxlabel_31")
        self.horizontalLayout_145.addWidget(self.cycxlabel_31)
        self.an_max_cap_txt = QtWidgets.QLineEdit(parent=self.dvdq)
        self.an_max_cap_txt.setMinimumSize(QtCore.QSize(310, 34))
        self.an_max_cap_txt.setMaximumSize(QtCore.QSize(310, 34))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.an_max_cap_txt.setFont(font)
        self.an_max_cap_txt.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhFormattedNumbersOnly)
        self.an_max_cap_txt.setText("")
        self.an_max_cap_txt.setObjectName("an_max_cap_txt")
        self.horizontalLayout_145.addWidget(self.an_max_cap_txt)
        self.verticalLayout_22.addLayout(self.horizontalLayout_145)
        self.line_13 = QtWidgets.QFrame(parent=self.dvdq)
        self.line_13.setMinimumSize(QtCore.QSize(656, 3))
        self.line_13.setMaximumSize(QtCore.QSize(656, 3))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.line_13.setFont(font)
        self.line_13.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        self.line_13.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.line_13.setObjectName("line_13")
        self.verticalLayout_22.addWidget(self.line_13)
        self.horizontalLayout_146 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_146.setObjectName("horizontalLayout_146")
        self.cycxlabel_18 = QtWidgets.QLabel(parent=self.dvdq)
        self.cycxlabel_18.setMinimumSize(QtCore.QSize(317, 33))
        self.cycxlabel_18.setMaximumSize(QtCore.QSize(317, 33))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.cycxlabel_18.setFont(font)
        self.cycxlabel_18.setObjectName("cycxlabel_18")
        self.horizontalLayout_146.addWidget(self.cycxlabel_18)
        self.ca_mass_ini_fix = QtWidgets.QCheckBox(parent=self.dvdq)
        self.ca_mass_ini_fix.setMinimumSize(QtCore.QSize(15, 13))
        self.ca_mass_ini_fix.setMaximumSize(QtCore.QSize(15, 13))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.ca_mass_ini_fix.setFont(font)
        self.ca_mass_ini_fix.setText("")
        self.ca_mass_ini_fix.setObjectName("ca_mass_ini_fix")
        self.horizontalLayout_146.addWidget(self.ca_mass_ini_fix)
        self.ca_mass_ini = QtWidgets.QLineEdit(parent=self.dvdq)
        self.ca_mass_ini.setMinimumSize(QtCore.QSize(310, 33))
        self.ca_mass_ini.setMaximumSize(QtCore.QSize(310, 33))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.ca_mass_ini.setFont(font)
        self.ca_mass_ini.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhFormattedNumbersOnly)
        self.ca_mass_ini.setText("")
        self.ca_mass_ini.setObjectName("ca_mass_ini")
        self.horizontalLayout_146.addWidget(self.ca_mass_ini)
        self.verticalLayout_22.addLayout(self.horizontalLayout_146)
        self.horizontalLayout_147 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_147.setObjectName("horizontalLayout_147")
        self.cycxlabel_23 = QtWidgets.QLabel(parent=self.dvdq)
        self.cycxlabel_23.setMinimumSize(QtCore.QSize(317, 34))
        self.cycxlabel_23.setMaximumSize(QtCore.QSize(317, 34))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.cycxlabel_23.setFont(font)
        self.cycxlabel_23.setObjectName("cycxlabel_23")
        self.horizontalLayout_147.addWidget(self.cycxlabel_23)
        self.ca_slip_ini_fix = QtWidgets.QCheckBox(parent=self.dvdq)
        self.ca_slip_ini_fix.setMinimumSize(QtCore.QSize(15, 13))
        self.ca_slip_ini_fix.setMaximumSize(QtCore.QSize(15, 13))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.ca_slip_ini_fix.setFont(font)
        self.ca_slip_ini_fix.setText("")
        self.ca_slip_ini_fix.setObjectName("ca_slip_ini_fix")
        self.horizontalLayout_147.addWidget(self.ca_slip_ini_fix)
        self.ca_slip_ini = QtWidgets.QLineEdit(parent=self.dvdq)
        self.ca_slip_ini.setMinimumSize(QtCore.QSize(310, 34))
        self.ca_slip_ini.setMaximumSize(QtCore.QSize(310, 34))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.ca_slip_ini.setFont(font)
        self.ca_slip_ini.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhFormattedNumbersOnly)
        self.ca_slip_ini.setText("")
        self.ca_slip_ini.setObjectName("ca_slip_ini")
        self.horizontalLayout_147.addWidget(self.ca_slip_ini)
        self.verticalLayout_22.addLayout(self.horizontalLayout_147)
        self.horizontalLayout_148 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_148.setObjectName("horizontalLayout_148")
        self.cycxlabel_24 = QtWidgets.QLabel(parent=self.dvdq)
        self.cycxlabel_24.setMinimumSize(QtCore.QSize(317, 33))
        self.cycxlabel_24.setMaximumSize(QtCore.QSize(317, 33))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.cycxlabel_24.setFont(font)
        self.cycxlabel_24.setObjectName("cycxlabel_24")
        self.horizontalLayout_148.addWidget(self.cycxlabel_24)
        self.an_mass_ini_fix = QtWidgets.QCheckBox(parent=self.dvdq)
        self.an_mass_ini_fix.setMinimumSize(QtCore.QSize(15, 13))
        self.an_mass_ini_fix.setMaximumSize(QtCore.QSize(15, 13))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.an_mass_ini_fix.setFont(font)
        self.an_mass_ini_fix.setText("")
        self.an_mass_ini_fix.setObjectName("an_mass_ini_fix")
        self.horizontalLayout_148.addWidget(self.an_mass_ini_fix)
        self.an_mass_ini = QtWidgets.QLineEdit(parent=self.dvdq)
        self.an_mass_ini.setMinimumSize(QtCore.QSize(310, 33))
        self.an_mass_ini.setMaximumSize(QtCore.QSize(310, 33))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.an_mass_ini.setFont(font)
        self.an_mass_ini.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhFormattedNumbersOnly)
        self.an_mass_ini.setText("")
        self.an_mass_ini.setObjectName("an_mass_ini")
        self.horizontalLayout_148.addWidget(self.an_mass_ini)
        self.verticalLayout_22.addLayout(self.horizontalLayout_148)
        self.horizontalLayout_149 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_149.setObjectName("horizontalLayout_149")
        self.cycxlabel_25 = QtWidgets.QLabel(parent=self.dvdq)
        self.cycxlabel_25.setMinimumSize(QtCore.QSize(317, 34))
        self.cycxlabel_25.setMaximumSize(QtCore.QSize(317, 34))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.cycxlabel_25.setFont(font)
        self.cycxlabel_25.setObjectName("cycxlabel_25")
        self.horizontalLayout_149.addWidget(self.cycxlabel_25)
        self.an_slip_ini_fix = QtWidgets.QCheckBox(parent=self.dvdq)
        self.an_slip_ini_fix.setMinimumSize(QtCore.QSize(15, 13))
        self.an_slip_ini_fix.setMaximumSize(QtCore.QSize(15, 13))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.an_slip_ini_fix.setFont(font)
        self.an_slip_ini_fix.setText("")
        self.an_slip_ini_fix.setObjectName("an_slip_ini_fix")
        self.horizontalLayout_149.addWidget(self.an_slip_ini_fix)
        self.an_slip_ini = QtWidgets.QLineEdit(parent=self.dvdq)
        self.an_slip_ini.setMinimumSize(QtCore.QSize(310, 34))
        self.an_slip_ini.setMaximumSize(QtCore.QSize(310, 34))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.an_slip_ini.setFont(font)
        self.an_slip_ini.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhFormattedNumbersOnly)
        self.an_slip_ini.setText("")
        self.an_slip_ini.setObjectName("an_slip_ini")
        self.horizontalLayout_149.addWidget(self.an_slip_ini)
        self.verticalLayout_22.addLayout(self.horizontalLayout_149)
        self.line_14 = QtWidgets.QFrame(parent=self.dvdq)
        self.line_14.setMinimumSize(QtCore.QSize(656, 3))
        self.line_14.setMaximumSize(QtCore.QSize(656, 3))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.line_14.setFont(font)
        self.line_14.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        self.line_14.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.line_14.setObjectName("line_14")
        self.verticalLayout_22.addWidget(self.line_14)
        self.horizontalLayout_150 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_150.setObjectName("horizontalLayout_150")
        self.cycxlabel_15 = QtWidgets.QLabel(parent=self.dvdq)
        self.cycxlabel_15.setMinimumSize(QtCore.QSize(338, 33))
        self.cycxlabel_15.setMaximumSize(QtCore.QSize(338, 33))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.cycxlabel_15.setFont(font)
        self.cycxlabel_15.setObjectName("cycxlabel_15")
        self.horizontalLayout_150.addWidget(self.cycxlabel_15)
        self.dvdq_test_no = QtWidgets.QLineEdit(parent=self.dvdq)
        self.dvdq_test_no.setMinimumSize(QtCore.QSize(310, 33))
        self.dvdq_test_no.setMaximumSize(QtCore.QSize(310, 33))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.dvdq_test_no.setFont(font)
        self.dvdq_test_no.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhFormattedNumbersOnly)
        self.dvdq_test_no.setObjectName("dvdq_test_no")
        self.horizontalLayout_150.addWidget(self.dvdq_test_no)
        self.verticalLayout_22.addLayout(self.horizontalLayout_150)
        self.horizontalLayout_151 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_151.setObjectName("horizontalLayout_151")
        self.cycxlabel_26 = QtWidgets.QLabel(parent=self.dvdq)
        self.cycxlabel_26.setMinimumSize(QtCore.QSize(338, 34))
        self.cycxlabel_26.setMaximumSize(QtCore.QSize(338, 34))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.cycxlabel_26.setFont(font)
        self.cycxlabel_26.setObjectName("cycxlabel_26")
        self.horizontalLayout_151.addWidget(self.cycxlabel_26)
        self.dvdq_rms = QtWidgets.QLineEdit(parent=self.dvdq)
        self.dvdq_rms.setMinimumSize(QtCore.QSize(310, 34))
        self.dvdq_rms.setMaximumSize(QtCore.QSize(310, 34))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.dvdq_rms.setFont(font)
        self.dvdq_rms.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhFormattedNumbersOnly)
        self.dvdq_rms.setText("")
        self.dvdq_rms.setObjectName("dvdq_rms")
        self.horizontalLayout_151.addWidget(self.dvdq_rms)
        self.verticalLayout_22.addLayout(self.horizontalLayout_151)
        self.line_15 = QtWidgets.QFrame(parent=self.dvdq)
        self.line_15.setMinimumSize(QtCore.QSize(656, 3))
        self.line_15.setMaximumSize(QtCore.QSize(656, 3))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.line_15.setFont(font)
        self.line_15.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        self.line_15.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.line_15.setObjectName("line_15")
        self.verticalLayout_22.addWidget(self.line_15)
        self.horizontalLayout_152 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_152.setObjectName("horizontalLayout_152")
        self.verticalLayout_15 = QtWidgets.QVBoxLayout()
        self.verticalLayout_15.setObjectName("verticalLayout_15")
        self.dvdq_ini_reset = QtWidgets.QPushButton(parent=self.dvdq)
        self.dvdq_ini_reset.setMinimumSize(QtCore.QSize(310, 50))
        self.dvdq_ini_reset.setMaximumSize(QtCore.QSize(310, 50))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.dvdq_ini_reset.setFont(font)
        self.dvdq_ini_reset.setObjectName("dvdq_ini_reset")
        self.verticalLayout_15.addWidget(self.dvdq_ini_reset)
        self.mat_dvdq_btn = QtWidgets.QPushButton(parent=self.dvdq)
        self.mat_dvdq_btn.setMinimumSize(QtCore.QSize(310, 50))
        self.mat_dvdq_btn.setMaximumSize(QtCore.QSize(310, 50))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.mat_dvdq_btn.setFont(font)
        self.mat_dvdq_btn.setObjectName("mat_dvdq_btn")
        self.verticalLayout_15.addWidget(self.mat_dvdq_btn)
        self.pro_dvdq_btn = QtWidgets.QPushButton(parent=self.dvdq)
        self.pro_dvdq_btn.setMinimumSize(QtCore.QSize(310, 50))
        self.pro_dvdq_btn.setMaximumSize(QtCore.QSize(310, 50))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.pro_dvdq_btn.setFont(font)
        self.pro_dvdq_btn.setObjectName("pro_dvdq_btn")
        self.verticalLayout_15.addWidget(self.pro_dvdq_btn)
        self.horizontalLayout_152.addLayout(self.verticalLayout_15)
        self.verticalLayout_16 = QtWidgets.QVBoxLayout()
        self.verticalLayout_16.setObjectName("verticalLayout_16")
        self.dvdq_tab_reset = QtWidgets.QPushButton(parent=self.dvdq)
        self.dvdq_tab_reset.setMinimumSize(QtCore.QSize(310, 50))
        self.dvdq_tab_reset.setMaximumSize(QtCore.QSize(310, 50))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.dvdq_tab_reset.setFont(font)
        self.dvdq_tab_reset.setObjectName("dvdq_tab_reset")
        self.verticalLayout_16.addWidget(self.dvdq_tab_reset)
        self.dvdq_fitting = QtWidgets.QPushButton(parent=self.dvdq)
        self.dvdq_fitting.setMinimumSize(QtCore.QSize(310, 50))
        self.dvdq_fitting.setMaximumSize(QtCore.QSize(310, 50))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.dvdq_fitting.setFont(font)
        self.dvdq_fitting.setObjectName("dvdq_fitting")
        self.verticalLayout_16.addWidget(self.dvdq_fitting)
        self.dvdq_fitting_2 = QtWidgets.QPushButton(parent=self.dvdq)
        self.dvdq_fitting_2.setMinimumSize(QtCore.QSize(310, 50))
        self.dvdq_fitting_2.setMaximumSize(QtCore.QSize(310, 50))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.dvdq_fitting_2.setFont(font)
        self.dvdq_fitting_2.setObjectName("dvdq_fitting_2")
        self.verticalLayout_16.addWidget(self.dvdq_fitting_2)
        self.horizontalLayout_152.addLayout(self.verticalLayout_16)
        self.verticalLayout_22.addLayout(self.horizontalLayout_152)
        self.horizontalLayout_153.addLayout(self.verticalLayout_22)
        self.dvdq_simul_tab = QtWidgets.QTabWidget(parent=self.dvdq)
        self.dvdq_simul_tab.setMinimumSize(QtCore.QSize(1200, 830))
        self.dvdq_simul_tab.setMaximumSize(QtCore.QSize(1200, 830))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.dvdq_simul_tab.setFont(font)
        self.dvdq_simul_tab.setObjectName("dvdq_simul_tab")
        self.horizontalLayout_153.addWidget(self.dvdq_simul_tab)
        self.horizontalLayout_154.addLayout(self.horizontalLayout_153)
        self.tabWidget.addTab(self.dvdq, "")
        self.tab_4 = QtWidgets.QWidget()
        self.tab_4.setObjectName("tab_4")
        self.horizontalLayout_158 = QtWidgets.QHBoxLayout(self.tab_4)
        self.horizontalLayout_158.setObjectName("horizontalLayout_158")
        self.horizontalLayout_157 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_157.setObjectName("horizontalLayout_157")
        self.verticalLayout_33 = QtWidgets.QVBoxLayout()
        self.verticalLayout_33.setObjectName("verticalLayout_33")
        self.horizontalLayout_93 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_93.setObjectName("horizontalLayout_93")
        self.cycxlabel_54 = QtWidgets.QLabel(parent=self.tab_4)
        self.cycxlabel_54.setMinimumSize(QtCore.QSize(300, 30))
        self.cycxlabel_54.setMaximumSize(QtCore.QSize(300, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.cycxlabel_54.setFont(font)
        self.cycxlabel_54.setObjectName("cycxlabel_54")
        self.horizontalLayout_93.addWidget(self.cycxlabel_54)
        self.TabReset_eu = QtWidgets.QPushButton(parent=self.tab_4)
        self.TabReset_eu.setMinimumSize(QtCore.QSize(300, 50))
        self.TabReset_eu.setMaximumSize(QtCore.QSize(300, 50))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(True)
        font.setItalic(False)
        font.setUnderline(True)
        font.setWeight(75)
        self.TabReset_eu.setFont(font)
        self.TabReset_eu.setObjectName("TabReset_eu")
        self.horizontalLayout_93.addWidget(self.TabReset_eu)
        self.verticalLayout_33.addLayout(self.horizontalLayout_93)
        self.cycparameter_eu = QtWidgets.QLineEdit(parent=self.tab_4)
        self.cycparameter_eu.setMinimumSize(QtCore.QSize(656, 23))
        self.cycparameter_eu.setMaximumSize(QtCore.QSize(656, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.cycparameter_eu.setFont(font)
        self.cycparameter_eu.setText("")
        self.cycparameter_eu.setObjectName("cycparameter_eu")
        self.verticalLayout_33.addWidget(self.cycparameter_eu)
        self.line_3 = QtWidgets.QFrame(parent=self.tab_4)
        self.line_3.setMinimumSize(QtCore.QSize(656, 3))
        self.line_3.setMaximumSize(QtCore.QSize(656, 3))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.line_3.setFont(font)
        self.line_3.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.line_3.setObjectName("line_3")
        self.verticalLayout_33.addWidget(self.line_3)
        self.horizontalLayout_156 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_156.setObjectName("horizontalLayout_156")
        self.aLabel_4 = QtWidgets.QLabel(parent=self.tab_4)
        self.aLabel_4.setMinimumSize(QtCore.QSize(200, 30))
        self.aLabel_4.setMaximumSize(QtCore.QSize(200, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.aLabel_4.setFont(font)
        self.aLabel_4.setObjectName("aLabel_4")
        self.horizontalLayout_156.addWidget(self.aLabel_4)
        self.fix_swelling_eu = QtWidgets.QCheckBox(parent=self.tab_4)
        self.fix_swelling_eu.setMinimumSize(QtCore.QSize(448, 18))
        self.fix_swelling_eu.setMaximumSize(QtCore.QSize(448, 18))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.fix_swelling_eu.setFont(font)
        self.fix_swelling_eu.setObjectName("fix_swelling_eu")
        self.horizontalLayout_156.addWidget(self.fix_swelling_eu)
        self.verticalLayout_33.addLayout(self.horizontalLayout_156)
        self.cycxlabel_38 = QtWidgets.QLabel(parent=self.tab_4)
        self.cycxlabel_38.setMinimumSize(QtCore.QSize(656, 30))
        self.cycxlabel_38.setMaximumSize(QtCore.QSize(656, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.cycxlabel_38.setFont(font)
        self.cycxlabel_38.setObjectName("cycxlabel_38")
        self.verticalLayout_33.addWidget(self.cycxlabel_38)
        self.horizontalLayout_94 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_94.setObjectName("horizontalLayout_94")
        self.aLabel_3 = QtWidgets.QLabel(parent=self.tab_4)
        self.aLabel_3.setMinimumSize(QtCore.QSize(150, 30))
        self.aLabel_3.setMaximumSize(QtCore.QSize(150, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.aLabel_3.setFont(font)
        self.aLabel_3.setObjectName("aLabel_3")
        self.horizontalLayout_94.addWidget(self.aLabel_3)
        self.aTextEdit_eu = QtWidgets.QLineEdit(parent=self.tab_4)
        self.aTextEdit_eu.setMinimumSize(QtCore.QSize(400, 23))
        self.aTextEdit_eu.setMaximumSize(QtCore.QSize(400, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.aTextEdit_eu.setFont(font)
        self.aTextEdit_eu.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhFormattedNumbersOnly)
        self.aTextEdit_eu.setText("")
        self.aTextEdit_eu.setObjectName("aTextEdit_eu")
        self.horizontalLayout_94.addWidget(self.aTextEdit_eu)
        self.verticalLayout_33.addLayout(self.horizontalLayout_94)
        self.horizontalLayout_95 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_95.setObjectName("horizontalLayout_95")
        self.bLabel_3 = QtWidgets.QLabel(parent=self.tab_4)
        self.bLabel_3.setMinimumSize(QtCore.QSize(150, 30))
        self.bLabel_3.setMaximumSize(QtCore.QSize(150, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.bLabel_3.setFont(font)
        self.bLabel_3.setObjectName("bLabel_3")
        self.horizontalLayout_95.addWidget(self.bLabel_3)
        self.bTextEdit_eu = QtWidgets.QLineEdit(parent=self.tab_4)
        self.bTextEdit_eu.setMinimumSize(QtCore.QSize(400, 23))
        self.bTextEdit_eu.setMaximumSize(QtCore.QSize(400, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.bTextEdit_eu.setFont(font)
        self.bTextEdit_eu.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhFormattedNumbersOnly)
        self.bTextEdit_eu.setText("")
        self.bTextEdit_eu.setObjectName("bTextEdit_eu")
        self.horizontalLayout_95.addWidget(self.bTextEdit_eu)
        self.verticalLayout_33.addLayout(self.horizontalLayout_95)
        self.horizontalLayout_96 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_96.setObjectName("horizontalLayout_96")
        self.b1Label_3 = QtWidgets.QLabel(parent=self.tab_4)
        self.b1Label_3.setMinimumSize(QtCore.QSize(150, 30))
        self.b1Label_3.setMaximumSize(QtCore.QSize(150, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.b1Label_3.setFont(font)
        self.b1Label_3.setObjectName("b1Label_3")
        self.horizontalLayout_96.addWidget(self.b1Label_3)
        self.b1TextEdit_eu = QtWidgets.QLineEdit(parent=self.tab_4)
        self.b1TextEdit_eu.setMinimumSize(QtCore.QSize(400, 23))
        self.b1TextEdit_eu.setMaximumSize(QtCore.QSize(400, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.b1TextEdit_eu.setFont(font)
        self.b1TextEdit_eu.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhFormattedNumbersOnly)
        self.b1TextEdit_eu.setText("")
        self.b1TextEdit_eu.setObjectName("b1TextEdit_eu")
        self.horizontalLayout_96.addWidget(self.b1TextEdit_eu)
        self.verticalLayout_33.addLayout(self.horizontalLayout_96)
        self.horizontalLayout_97 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_97.setObjectName("horizontalLayout_97")
        self.cLabel_3 = QtWidgets.QLabel(parent=self.tab_4)
        self.cLabel_3.setMinimumSize(QtCore.QSize(150, 30))
        self.cLabel_3.setMaximumSize(QtCore.QSize(150, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.cLabel_3.setFont(font)
        self.cLabel_3.setObjectName("cLabel_3")
        self.horizontalLayout_97.addWidget(self.cLabel_3)
        self.cTextEdit_eu = QtWidgets.QLineEdit(parent=self.tab_4)
        self.cTextEdit_eu.setMinimumSize(QtCore.QSize(400, 23))
        self.cTextEdit_eu.setMaximumSize(QtCore.QSize(400, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.cTextEdit_eu.setFont(font)
        self.cTextEdit_eu.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhFormattedNumbersOnly)
        self.cTextEdit_eu.setText("")
        self.cTextEdit_eu.setObjectName("cTextEdit_eu")
        self.horizontalLayout_97.addWidget(self.cTextEdit_eu)
        self.verticalLayout_33.addLayout(self.horizontalLayout_97)
        self.horizontalLayout_98 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_98.setObjectName("horizontalLayout_98")
        self.dLabel_3 = QtWidgets.QLabel(parent=self.tab_4)
        self.dLabel_3.setMinimumSize(QtCore.QSize(150, 29))
        self.dLabel_3.setMaximumSize(QtCore.QSize(150, 29))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.dLabel_3.setFont(font)
        self.dLabel_3.setObjectName("dLabel_3")
        self.horizontalLayout_98.addWidget(self.dLabel_3)
        self.dTextEdit_eu = QtWidgets.QLineEdit(parent=self.tab_4)
        self.dTextEdit_eu.setMinimumSize(QtCore.QSize(400, 23))
        self.dTextEdit_eu.setMaximumSize(QtCore.QSize(400, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.dTextEdit_eu.setFont(font)
        self.dTextEdit_eu.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhFormattedNumbersOnly)
        self.dTextEdit_eu.setText("")
        self.dTextEdit_eu.setObjectName("dTextEdit_eu")
        self.horizontalLayout_98.addWidget(self.dTextEdit_eu)
        self.verticalLayout_33.addLayout(self.horizontalLayout_98)
        self.horizontalLayout_99 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_99.setObjectName("horizontalLayout_99")
        self.eLabel_3 = QtWidgets.QLabel(parent=self.tab_4)
        self.eLabel_3.setMinimumSize(QtCore.QSize(150, 30))
        self.eLabel_3.setMaximumSize(QtCore.QSize(150, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.eLabel_3.setFont(font)
        self.eLabel_3.setObjectName("eLabel_3")
        self.horizontalLayout_99.addWidget(self.eLabel_3)
        self.eTextEdit_eu = QtWidgets.QLineEdit(parent=self.tab_4)
        self.eTextEdit_eu.setMinimumSize(QtCore.QSize(400, 23))
        self.eTextEdit_eu.setMaximumSize(QtCore.QSize(400, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.eTextEdit_eu.setFont(font)
        self.eTextEdit_eu.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhFormattedNumbersOnly)
        self.eTextEdit_eu.setText("")
        self.eTextEdit_eu.setObjectName("eTextEdit_eu")
        self.horizontalLayout_99.addWidget(self.eTextEdit_eu)
        self.verticalLayout_33.addLayout(self.horizontalLayout_99)
        self.horizontalLayout_100 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_100.setObjectName("horizontalLayout_100")
        self.fLabel_3 = QtWidgets.QLabel(parent=self.tab_4)
        self.fLabel_3.setMinimumSize(QtCore.QSize(150, 30))
        self.fLabel_3.setMaximumSize(QtCore.QSize(150, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.fLabel_3.setFont(font)
        self.fLabel_3.setObjectName("fLabel_3")
        self.horizontalLayout_100.addWidget(self.fLabel_3)
        self.fTextEdit_eu = QtWidgets.QLineEdit(parent=self.tab_4)
        self.fTextEdit_eu.setMinimumSize(QtCore.QSize(400, 23))
        self.fTextEdit_eu.setMaximumSize(QtCore.QSize(400, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.fTextEdit_eu.setFont(font)
        self.fTextEdit_eu.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhFormattedNumbersOnly)
        self.fTextEdit_eu.setText("")
        self.fTextEdit_eu.setObjectName("fTextEdit_eu")
        self.horizontalLayout_100.addWidget(self.fTextEdit_eu)
        self.verticalLayout_33.addLayout(self.horizontalLayout_100)
        self.horizontalLayout_101 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_101.setObjectName("horizontalLayout_101")
        self.fdLabel_3 = QtWidgets.QLabel(parent=self.tab_4)
        self.fdLabel_3.setMinimumSize(QtCore.QSize(150, 30))
        self.fdLabel_3.setMaximumSize(QtCore.QSize(150, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.fdLabel_3.setFont(font)
        self.fdLabel_3.setObjectName("fdLabel_3")
        self.horizontalLayout_101.addWidget(self.fdLabel_3)
        self.fdTextEdit_eu = QtWidgets.QLineEdit(parent=self.tab_4)
        self.fdTextEdit_eu.setMinimumSize(QtCore.QSize(400, 23))
        self.fdTextEdit_eu.setMaximumSize(QtCore.QSize(400, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.fdTextEdit_eu.setFont(font)
        self.fdTextEdit_eu.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhFormattedNumbersOnly)
        self.fdTextEdit_eu.setText("")
        self.fdTextEdit_eu.setObjectName("fdTextEdit_eu")
        self.horizontalLayout_101.addWidget(self.fdTextEdit_eu)
        self.verticalLayout_33.addLayout(self.horizontalLayout_101)
        self.horizontalLayout_128 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_128.setObjectName("horizontalLayout_128")
        self.fdLabel_4 = QtWidgets.QLabel(parent=self.tab_4)
        self.fdLabel_4.setMinimumSize(QtCore.QSize(150, 30))
        self.fdLabel_4.setMaximumSize(QtCore.QSize(150, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.fdLabel_4.setFont(font)
        self.fdLabel_4.setObjectName("fdLabel_4")
        self.horizontalLayout_128.addWidget(self.fdLabel_4)
        self.fdTextEdit_eu_2 = QtWidgets.QLineEdit(parent=self.tab_4)
        self.fdTextEdit_eu_2.setMinimumSize(QtCore.QSize(400, 23))
        self.fdTextEdit_eu_2.setMaximumSize(QtCore.QSize(400, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.fdTextEdit_eu_2.setFont(font)
        self.fdTextEdit_eu_2.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhFormattedNumbersOnly)
        self.fdTextEdit_eu_2.setText("")
        self.fdTextEdit_eu_2.setObjectName("fdTextEdit_eu_2")
        self.horizontalLayout_128.addWidget(self.fdTextEdit_eu_2)
        self.verticalLayout_33.addLayout(self.horizontalLayout_128)
        self.horizontalLayout_102 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_102.setObjectName("horizontalLayout_102")
        self.cycxlabel_36 = QtWidgets.QLabel(parent=self.tab_4)
        self.cycxlabel_36.setMinimumSize(QtCore.QSize(150, 30))
        self.cycxlabel_36.setMaximumSize(QtCore.QSize(150, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.cycxlabel_36.setFont(font)
        self.cycxlabel_36.setObjectName("cycxlabel_36")
        self.horizontalLayout_102.addWidget(self.cycxlabel_36)
        self.simul_y_max_eu = QtWidgets.QLineEdit(parent=self.tab_4)
        self.simul_y_max_eu.setMinimumSize(QtCore.QSize(400, 23))
        self.simul_y_max_eu.setMaximumSize(QtCore.QSize(400, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.simul_y_max_eu.setFont(font)
        self.simul_y_max_eu.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhFormattedNumbersOnly)
        self.simul_y_max_eu.setObjectName("simul_y_max_eu")
        self.horizontalLayout_102.addWidget(self.simul_y_max_eu)
        self.verticalLayout_33.addLayout(self.horizontalLayout_102)
        self.horizontalLayout_103 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_103.setObjectName("horizontalLayout_103")
        self.cycxlabel_35 = QtWidgets.QLabel(parent=self.tab_4)
        self.cycxlabel_35.setMinimumSize(QtCore.QSize(150, 30))
        self.cycxlabel_35.setMaximumSize(QtCore.QSize(150, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.cycxlabel_35.setFont(font)
        self.cycxlabel_35.setObjectName("cycxlabel_35")
        self.horizontalLayout_103.addWidget(self.cycxlabel_35)
        self.simul_y_min_eu = QtWidgets.QLineEdit(parent=self.tab_4)
        self.simul_y_min_eu.setMinimumSize(QtCore.QSize(400, 23))
        self.simul_y_min_eu.setMaximumSize(QtCore.QSize(400, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.simul_y_min_eu.setFont(font)
        self.simul_y_min_eu.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhFormattedNumbersOnly)
        self.simul_y_min_eu.setObjectName("simul_y_min_eu")
        self.horizontalLayout_103.addWidget(self.simul_y_min_eu)
        self.verticalLayout_33.addLayout(self.horizontalLayout_103)
        self.horizontalLayout_104 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_104.setObjectName("horizontalLayout_104")
        self.cycxlabel_37 = QtWidgets.QLabel(parent=self.tab_4)
        self.cycxlabel_37.setMinimumSize(QtCore.QSize(150, 30))
        self.cycxlabel_37.setMaximumSize(QtCore.QSize(150, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.cycxlabel_37.setFont(font)
        self.cycxlabel_37.setObjectName("cycxlabel_37")
        self.horizontalLayout_104.addWidget(self.cycxlabel_37)
        self.simul_x_max_eu = QtWidgets.QLineEdit(parent=self.tab_4)
        self.simul_x_max_eu.setMinimumSize(QtCore.QSize(400, 23))
        self.simul_x_max_eu.setMaximumSize(QtCore.QSize(400, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.simul_x_max_eu.setFont(font)
        self.simul_x_max_eu.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhDigitsOnly)
        self.simul_x_max_eu.setObjectName("simul_x_max_eu")
        self.horizontalLayout_104.addWidget(self.simul_x_max_eu)
        self.verticalLayout_33.addLayout(self.horizontalLayout_104)
        self.horizontalLayout_105 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_105.setObjectName("horizontalLayout_105")
        self.ParameterReset_eu = QtWidgets.QPushButton(parent=self.tab_4)
        self.ParameterReset_eu.setMinimumSize(QtCore.QSize(205, 80))
        self.ParameterReset_eu.setMaximumSize(QtCore.QSize(205, 80))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.ParameterReset_eu.setFont(font)
        self.ParameterReset_eu.setObjectName("ParameterReset_eu")
        self.horizontalLayout_105.addWidget(self.ParameterReset_eu)
        self.load_cycparameter_eu = QtWidgets.QPushButton(parent=self.tab_4)
        self.load_cycparameter_eu.setMinimumSize(QtCore.QSize(205, 80))
        self.load_cycparameter_eu.setMaximumSize(QtCore.QSize(205, 80))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.load_cycparameter_eu.setFont(font)
        self.load_cycparameter_eu.setObjectName("load_cycparameter_eu")
        self.horizontalLayout_105.addWidget(self.load_cycparameter_eu)
        self.save_cycparameter_eu = QtWidgets.QPushButton(parent=self.tab_4)
        self.save_cycparameter_eu.setMinimumSize(QtCore.QSize(205, 80))
        self.save_cycparameter_eu.setMaximumSize(QtCore.QSize(205, 80))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.save_cycparameter_eu.setFont(font)
        self.save_cycparameter_eu.setObjectName("save_cycparameter_eu")
        self.horizontalLayout_105.addWidget(self.save_cycparameter_eu)
        self.verticalLayout_33.addLayout(self.horizontalLayout_105)
        self.horizontalLayout_106 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_106.setObjectName("horizontalLayout_106")
        self.FitConfirm_eu = QtWidgets.QPushButton(parent=self.tab_4)
        self.FitConfirm_eu.setMinimumSize(QtCore.QSize(205, 80))
        self.FitConfirm_eu.setMaximumSize(QtCore.QSize(205, 80))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.FitConfirm_eu.setFont(font)
        self.FitConfirm_eu.setObjectName("FitConfirm_eu")
        self.horizontalLayout_106.addWidget(self.FitConfirm_eu)
        self.ConstFitConfirm_eu = QtWidgets.QPushButton(parent=self.tab_4)
        self.ConstFitConfirm_eu.setMinimumSize(QtCore.QSize(205, 80))
        self.ConstFitConfirm_eu.setMaximumSize(QtCore.QSize(205, 80))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.ConstFitConfirm_eu.setFont(font)
        self.ConstFitConfirm_eu.setObjectName("ConstFitConfirm_eu")
        self.horizontalLayout_106.addWidget(self.ConstFitConfirm_eu)
        self.indivConstFitConfirm_eu = QtWidgets.QPushButton(parent=self.tab_4)
        self.indivConstFitConfirm_eu.setMinimumSize(QtCore.QSize(205, 80))
        self.indivConstFitConfirm_eu.setMaximumSize(QtCore.QSize(205, 80))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.indivConstFitConfirm_eu.setFont(font)
        self.indivConstFitConfirm_eu.setObjectName("indivConstFitConfirm_eu")
        self.horizontalLayout_106.addWidget(self.indivConstFitConfirm_eu)
        self.verticalLayout_33.addLayout(self.horizontalLayout_106)
        self.horizontalLayout_157.addLayout(self.verticalLayout_33)
        self.cycle_simul_tab_eu = QtWidgets.QTabWidget(parent=self.tab_4)
        self.cycle_simul_tab_eu.setMinimumSize(QtCore.QSize(1200, 830))
        self.cycle_simul_tab_eu.setMaximumSize(QtCore.QSize(1200, 830))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.cycle_simul_tab_eu.setFont(font)
        self.cycle_simul_tab_eu.setObjectName("cycle_simul_tab_eu")
        self.horizontalLayout_157.addWidget(self.cycle_simul_tab_eu)
        self.horizontalLayout_158.addLayout(self.horizontalLayout_157)
        self.tabWidget.addTab(self.tab_4, "")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.horizontalLayout_169 = QtWidgets.QHBoxLayout(self.tab_3)
        self.horizontalLayout_169.setObjectName("horizontalLayout_169")
        self.horizontalLayout_168 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_168.setObjectName("horizontalLayout_168")
        self.verticalLayout_36 = QtWidgets.QVBoxLayout()
        self.verticalLayout_36.setObjectName("verticalLayout_36")
        self.horizontalLayout_159 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_159.setObjectName("horizontalLayout_159")
        self.cycxlabel_52 = QtWidgets.QLabel(parent=self.tab_3)
        self.cycxlabel_52.setMinimumSize(QtCore.QSize(150, 25))
        self.cycxlabel_52.setMaximumSize(QtCore.QSize(150, 25))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.cycxlabel_52.setFont(font)
        self.cycxlabel_52.setObjectName("cycxlabel_52")
        self.horizontalLayout_159.addWidget(self.cycxlabel_52)
        self.cycparameter = QtWidgets.QLineEdit(parent=self.tab_3)
        self.cycparameter.setMinimumSize(QtCore.QSize(450, 25))
        self.cycparameter.setMaximumSize(QtCore.QSize(450, 25))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.cycparameter.setFont(font)
        self.cycparameter.setText("")
        self.cycparameter.setObjectName("cycparameter")
        self.horizontalLayout_159.addWidget(self.cycparameter)
        self.verticalLayout_36.addLayout(self.horizontalLayout_159)
        self.horizontalLayout_160 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_160.setObjectName("horizontalLayout_160")
        self.cycxlabel_55 = QtWidgets.QLabel(parent=self.tab_3)
        self.cycxlabel_55.setMinimumSize(QtCore.QSize(150, 25))
        self.cycxlabel_55.setMaximumSize(QtCore.QSize(150, 25))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.cycxlabel_55.setFont(font)
        self.cycxlabel_55.setObjectName("cycxlabel_55")
        self.horizontalLayout_160.addWidget(self.cycxlabel_55)
        self.cycparameter2 = QtWidgets.QLineEdit(parent=self.tab_3)
        self.cycparameter2.setMinimumSize(QtCore.QSize(450, 25))
        self.cycparameter2.setMaximumSize(QtCore.QSize(450, 25))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.cycparameter2.setFont(font)
        self.cycparameter2.setText("")
        self.cycparameter2.setObjectName("cycparameter2")
        self.horizontalLayout_160.addWidget(self.cycparameter2)
        self.verticalLayout_36.addLayout(self.horizontalLayout_160)
        self.horizontalLayout_161 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_161.setObjectName("horizontalLayout_161")
        self.FitGroupConst_5 = QtWidgets.QGroupBox(parent=self.tab_3)
        self.FitGroupConst_5.setMinimumSize(QtCore.QSize(315, 400))
        self.FitGroupConst_5.setMaximumSize(QtCore.QSize(315, 400))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.FitGroupConst_5.setFont(font)
        self.FitGroupConst_5.setObjectName("FitGroupConst_5")
        self.gridLayout_10 = QtWidgets.QGridLayout(self.FitGroupConst_5)
        self.gridLayout_10.setObjectName("gridLayout_10")
        self.horizontalLayout_32 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_32.setObjectName("horizontalLayout_32")
        self.aLabel_2 = QtWidgets.QLabel(parent=self.FitGroupConst_5)
        self.aLabel_2.setMinimumSize(QtCore.QSize(50, 30))
        self.aLabel_2.setMaximumSize(QtCore.QSize(50, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.aLabel_2.setFont(font)
        self.aLabel_2.setObjectName("aLabel_2")
        self.horizontalLayout_32.addWidget(self.aLabel_2)
        self.aTextEdit_02c = QtWidgets.QLineEdit(parent=self.FitGroupConst_5)
        self.aTextEdit_02c.setMinimumSize(QtCore.QSize(200, 23))
        self.aTextEdit_02c.setMaximumSize(QtCore.QSize(200, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.aTextEdit_02c.setFont(font)
        self.aTextEdit_02c.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhFormattedNumbersOnly)
        self.aTextEdit_02c.setText("")
        self.aTextEdit_02c.setObjectName("aTextEdit_02c")
        self.horizontalLayout_32.addWidget(self.aTextEdit_02c)
        self.gridLayout_10.addLayout(self.horizontalLayout_32, 0, 0, 1, 1)
        self.horizontalLayout_33 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_33.setObjectName("horizontalLayout_33")
        self.bLabel_2 = QtWidgets.QLabel(parent=self.FitGroupConst_5)
        self.bLabel_2.setMinimumSize(QtCore.QSize(50, 30))
        self.bLabel_2.setMaximumSize(QtCore.QSize(50, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.bLabel_2.setFont(font)
        self.bLabel_2.setObjectName("bLabel_2")
        self.horizontalLayout_33.addWidget(self.bLabel_2)
        self.bTextEdit_02c = QtWidgets.QLineEdit(parent=self.FitGroupConst_5)
        self.bTextEdit_02c.setMinimumSize(QtCore.QSize(200, 23))
        self.bTextEdit_02c.setMaximumSize(QtCore.QSize(200, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.bTextEdit_02c.setFont(font)
        self.bTextEdit_02c.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhFormattedNumbersOnly)
        self.bTextEdit_02c.setText("")
        self.bTextEdit_02c.setObjectName("bTextEdit_02c")
        self.horizontalLayout_33.addWidget(self.bTextEdit_02c)
        self.gridLayout_10.addLayout(self.horizontalLayout_33, 1, 0, 1, 1)
        self.horizontalLayout_35 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_35.setObjectName("horizontalLayout_35")
        self.b1Label_2 = QtWidgets.QLabel(parent=self.FitGroupConst_5)
        self.b1Label_2.setMinimumSize(QtCore.QSize(50, 30))
        self.b1Label_2.setMaximumSize(QtCore.QSize(50, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.b1Label_2.setFont(font)
        self.b1Label_2.setObjectName("b1Label_2")
        self.horizontalLayout_35.addWidget(self.b1Label_2)
        self.b1TextEdit_02c = QtWidgets.QLineEdit(parent=self.FitGroupConst_5)
        self.b1TextEdit_02c.setMinimumSize(QtCore.QSize(200, 23))
        self.b1TextEdit_02c.setMaximumSize(QtCore.QSize(200, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.b1TextEdit_02c.setFont(font)
        self.b1TextEdit_02c.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhFormattedNumbersOnly)
        self.b1TextEdit_02c.setText("")
        self.b1TextEdit_02c.setObjectName("b1TextEdit_02c")
        self.horizontalLayout_35.addWidget(self.b1TextEdit_02c)
        self.gridLayout_10.addLayout(self.horizontalLayout_35, 2, 0, 1, 1)
        self.horizontalLayout_63 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_63.setObjectName("horizontalLayout_63")
        self.cLabel_2 = QtWidgets.QLabel(parent=self.FitGroupConst_5)
        self.cLabel_2.setMinimumSize(QtCore.QSize(50, 30))
        self.cLabel_2.setMaximumSize(QtCore.QSize(50, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.cLabel_2.setFont(font)
        self.cLabel_2.setObjectName("cLabel_2")
        self.horizontalLayout_63.addWidget(self.cLabel_2)
        self.cTextEdit_02c = QtWidgets.QLineEdit(parent=self.FitGroupConst_5)
        self.cTextEdit_02c.setMinimumSize(QtCore.QSize(200, 23))
        self.cTextEdit_02c.setMaximumSize(QtCore.QSize(200, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.cTextEdit_02c.setFont(font)
        self.cTextEdit_02c.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhFormattedNumbersOnly)
        self.cTextEdit_02c.setText("")
        self.cTextEdit_02c.setObjectName("cTextEdit_02c")
        self.horizontalLayout_63.addWidget(self.cTextEdit_02c)
        self.gridLayout_10.addLayout(self.horizontalLayout_63, 3, 0, 1, 1)
        self.horizontalLayout_64 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_64.setObjectName("horizontalLayout_64")
        self.dLabel_2 = QtWidgets.QLabel(parent=self.FitGroupConst_5)
        self.dLabel_2.setMinimumSize(QtCore.QSize(50, 29))
        self.dLabel_2.setMaximumSize(QtCore.QSize(50, 29))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.dLabel_2.setFont(font)
        self.dLabel_2.setObjectName("dLabel_2")
        self.horizontalLayout_64.addWidget(self.dLabel_2)
        self.dTextEdit_02c = QtWidgets.QLineEdit(parent=self.FitGroupConst_5)
        self.dTextEdit_02c.setMinimumSize(QtCore.QSize(200, 23))
        self.dTextEdit_02c.setMaximumSize(QtCore.QSize(200, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.dTextEdit_02c.setFont(font)
        self.dTextEdit_02c.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhFormattedNumbersOnly)
        self.dTextEdit_02c.setText("")
        self.dTextEdit_02c.setObjectName("dTextEdit_02c")
        self.horizontalLayout_64.addWidget(self.dTextEdit_02c)
        self.gridLayout_10.addLayout(self.horizontalLayout_64, 4, 0, 1, 1)
        self.horizontalLayout_65 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_65.setObjectName("horizontalLayout_65")
        self.eLabel_2 = QtWidgets.QLabel(parent=self.FitGroupConst_5)
        self.eLabel_2.setMinimumSize(QtCore.QSize(50, 30))
        self.eLabel_2.setMaximumSize(QtCore.QSize(50, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.eLabel_2.setFont(font)
        self.eLabel_2.setObjectName("eLabel_2")
        self.horizontalLayout_65.addWidget(self.eLabel_2)
        self.eTextEdit_02c = QtWidgets.QLineEdit(parent=self.FitGroupConst_5)
        self.eTextEdit_02c.setMinimumSize(QtCore.QSize(200, 23))
        self.eTextEdit_02c.setMaximumSize(QtCore.QSize(200, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.eTextEdit_02c.setFont(font)
        self.eTextEdit_02c.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhFormattedNumbersOnly)
        self.eTextEdit_02c.setText("")
        self.eTextEdit_02c.setObjectName("eTextEdit_02c")
        self.horizontalLayout_65.addWidget(self.eTextEdit_02c)
        self.gridLayout_10.addLayout(self.horizontalLayout_65, 5, 0, 1, 1)
        self.horizontalLayout_66 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_66.setObjectName("horizontalLayout_66")
        self.fLabel_2 = QtWidgets.QLabel(parent=self.FitGroupConst_5)
        self.fLabel_2.setMinimumSize(QtCore.QSize(50, 30))
        self.fLabel_2.setMaximumSize(QtCore.QSize(50, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.fLabel_2.setFont(font)
        self.fLabel_2.setObjectName("fLabel_2")
        self.horizontalLayout_66.addWidget(self.fLabel_2)
        self.fTextEdit_02c = QtWidgets.QLineEdit(parent=self.FitGroupConst_5)
        self.fTextEdit_02c.setMinimumSize(QtCore.QSize(200, 23))
        self.fTextEdit_02c.setMaximumSize(QtCore.QSize(200, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.fTextEdit_02c.setFont(font)
        self.fTextEdit_02c.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhFormattedNumbersOnly)
        self.fTextEdit_02c.setText("")
        self.fTextEdit_02c.setObjectName("fTextEdit_02c")
        self.horizontalLayout_66.addWidget(self.fTextEdit_02c)
        self.gridLayout_10.addLayout(self.horizontalLayout_66, 6, 0, 1, 1)
        self.horizontalLayout_67 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_67.setObjectName("horizontalLayout_67")
        self.fdLabel_2 = QtWidgets.QLabel(parent=self.FitGroupConst_5)
        self.fdLabel_2.setMinimumSize(QtCore.QSize(50, 30))
        self.fdLabel_2.setMaximumSize(QtCore.QSize(50, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.fdLabel_2.setFont(font)
        self.fdLabel_2.setObjectName("fdLabel_2")
        self.horizontalLayout_67.addWidget(self.fdLabel_2)
        self.fdTextEdit_02c = QtWidgets.QLineEdit(parent=self.FitGroupConst_5)
        self.fdTextEdit_02c.setMinimumSize(QtCore.QSize(200, 23))
        self.fdTextEdit_02c.setMaximumSize(QtCore.QSize(200, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.fdTextEdit_02c.setFont(font)
        self.fdTextEdit_02c.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhFormattedNumbersOnly)
        self.fdTextEdit_02c.setText("")
        self.fdTextEdit_02c.setObjectName("fdTextEdit_02c")
        self.horizontalLayout_67.addWidget(self.fdTextEdit_02c)
        self.gridLayout_10.addLayout(self.horizontalLayout_67, 7, 0, 1, 1)
        self.horizontalLayout_161.addWidget(self.FitGroupConst_5)
        self.FitGroupConst_6 = QtWidgets.QGroupBox(parent=self.tab_3)
        self.FitGroupConst_6.setMinimumSize(QtCore.QSize(315, 400))
        self.FitGroupConst_6.setMaximumSize(QtCore.QSize(315, 400))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.FitGroupConst_6.setFont(font)
        self.FitGroupConst_6.setObjectName("FitGroupConst_6")
        self.gridLayout_15 = QtWidgets.QGridLayout(self.FitGroupConst_6)
        self.gridLayout_15.setObjectName("gridLayout_15")
        self.horizontalLayout_68 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_68.setObjectName("horizontalLayout_68")
        self.aLabel_8 = QtWidgets.QLabel(parent=self.FitGroupConst_6)
        self.aLabel_8.setMinimumSize(QtCore.QSize(50, 30))
        self.aLabel_8.setMaximumSize(QtCore.QSize(50, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.aLabel_8.setFont(font)
        self.aLabel_8.setObjectName("aLabel_8")
        self.horizontalLayout_68.addWidget(self.aLabel_8)
        self.aTextEdit_05c = QtWidgets.QLineEdit(parent=self.FitGroupConst_6)
        self.aTextEdit_05c.setMinimumSize(QtCore.QSize(200, 23))
        self.aTextEdit_05c.setMaximumSize(QtCore.QSize(200, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.aTextEdit_05c.setFont(font)
        self.aTextEdit_05c.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhFormattedNumbersOnly)
        self.aTextEdit_05c.setText("")
        self.aTextEdit_05c.setObjectName("aTextEdit_05c")
        self.horizontalLayout_68.addWidget(self.aTextEdit_05c)
        self.gridLayout_15.addLayout(self.horizontalLayout_68, 0, 0, 1, 1)
        self.horizontalLayout_69 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_69.setObjectName("horizontalLayout_69")
        self.bLabel_8 = QtWidgets.QLabel(parent=self.FitGroupConst_6)
        self.bLabel_8.setMinimumSize(QtCore.QSize(50, 30))
        self.bLabel_8.setMaximumSize(QtCore.QSize(50, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.bLabel_8.setFont(font)
        self.bLabel_8.setObjectName("bLabel_8")
        self.horizontalLayout_69.addWidget(self.bLabel_8)
        self.bTextEdit_05c = QtWidgets.QLineEdit(parent=self.FitGroupConst_6)
        self.bTextEdit_05c.setMinimumSize(QtCore.QSize(200, 23))
        self.bTextEdit_05c.setMaximumSize(QtCore.QSize(200, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.bTextEdit_05c.setFont(font)
        self.bTextEdit_05c.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhFormattedNumbersOnly)
        self.bTextEdit_05c.setText("")
        self.bTextEdit_05c.setObjectName("bTextEdit_05c")
        self.horizontalLayout_69.addWidget(self.bTextEdit_05c)
        self.gridLayout_15.addLayout(self.horizontalLayout_69, 1, 0, 1, 1)
        self.horizontalLayout_74 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_74.setObjectName("horizontalLayout_74")
        self.b1Label_8 = QtWidgets.QLabel(parent=self.FitGroupConst_6)
        self.b1Label_8.setMinimumSize(QtCore.QSize(50, 30))
        self.b1Label_8.setMaximumSize(QtCore.QSize(50, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.b1Label_8.setFont(font)
        self.b1Label_8.setObjectName("b1Label_8")
        self.horizontalLayout_74.addWidget(self.b1Label_8)
        self.b1TextEdit_05c = QtWidgets.QLineEdit(parent=self.FitGroupConst_6)
        self.b1TextEdit_05c.setMinimumSize(QtCore.QSize(200, 23))
        self.b1TextEdit_05c.setMaximumSize(QtCore.QSize(200, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.b1TextEdit_05c.setFont(font)
        self.b1TextEdit_05c.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhFormattedNumbersOnly)
        self.b1TextEdit_05c.setText("")
        self.b1TextEdit_05c.setObjectName("b1TextEdit_05c")
        self.horizontalLayout_74.addWidget(self.b1TextEdit_05c)
        self.gridLayout_15.addLayout(self.horizontalLayout_74, 2, 0, 1, 1)
        self.horizontalLayout_75 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_75.setObjectName("horizontalLayout_75")
        self.cLabel_8 = QtWidgets.QLabel(parent=self.FitGroupConst_6)
        self.cLabel_8.setMinimumSize(QtCore.QSize(50, 30))
        self.cLabel_8.setMaximumSize(QtCore.QSize(50, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.cLabel_8.setFont(font)
        self.cLabel_8.setObjectName("cLabel_8")
        self.horizontalLayout_75.addWidget(self.cLabel_8)
        self.cTextEdit_05c = QtWidgets.QLineEdit(parent=self.FitGroupConst_6)
        self.cTextEdit_05c.setMinimumSize(QtCore.QSize(200, 23))
        self.cTextEdit_05c.setMaximumSize(QtCore.QSize(200, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.cTextEdit_05c.setFont(font)
        self.cTextEdit_05c.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhFormattedNumbersOnly)
        self.cTextEdit_05c.setText("")
        self.cTextEdit_05c.setObjectName("cTextEdit_05c")
        self.horizontalLayout_75.addWidget(self.cTextEdit_05c)
        self.gridLayout_15.addLayout(self.horizontalLayout_75, 3, 0, 1, 1)
        self.horizontalLayout_76 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_76.setObjectName("horizontalLayout_76")
        self.dLabel_8 = QtWidgets.QLabel(parent=self.FitGroupConst_6)
        self.dLabel_8.setMinimumSize(QtCore.QSize(50, 29))
        self.dLabel_8.setMaximumSize(QtCore.QSize(50, 29))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.dLabel_8.setFont(font)
        self.dLabel_8.setObjectName("dLabel_8")
        self.horizontalLayout_76.addWidget(self.dLabel_8)
        self.dTextEdit_05c = QtWidgets.QLineEdit(parent=self.FitGroupConst_6)
        self.dTextEdit_05c.setMinimumSize(QtCore.QSize(200, 23))
        self.dTextEdit_05c.setMaximumSize(QtCore.QSize(200, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.dTextEdit_05c.setFont(font)
        self.dTextEdit_05c.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhFormattedNumbersOnly)
        self.dTextEdit_05c.setText("")
        self.dTextEdit_05c.setObjectName("dTextEdit_05c")
        self.horizontalLayout_76.addWidget(self.dTextEdit_05c)
        self.gridLayout_15.addLayout(self.horizontalLayout_76, 4, 0, 1, 1)
        self.horizontalLayout_77 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_77.setObjectName("horizontalLayout_77")
        self.eLabel_8 = QtWidgets.QLabel(parent=self.FitGroupConst_6)
        self.eLabel_8.setMinimumSize(QtCore.QSize(50, 30))
        self.eLabel_8.setMaximumSize(QtCore.QSize(50, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.eLabel_8.setFont(font)
        self.eLabel_8.setObjectName("eLabel_8")
        self.horizontalLayout_77.addWidget(self.eLabel_8)
        self.eTextEdit_05c = QtWidgets.QLineEdit(parent=self.FitGroupConst_6)
        self.eTextEdit_05c.setMinimumSize(QtCore.QSize(200, 23))
        self.eTextEdit_05c.setMaximumSize(QtCore.QSize(200, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.eTextEdit_05c.setFont(font)
        self.eTextEdit_05c.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhFormattedNumbersOnly)
        self.eTextEdit_05c.setText("")
        self.eTextEdit_05c.setObjectName("eTextEdit_05c")
        self.horizontalLayout_77.addWidget(self.eTextEdit_05c)
        self.gridLayout_15.addLayout(self.horizontalLayout_77, 5, 0, 1, 1)
        self.horizontalLayout_78 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_78.setObjectName("horizontalLayout_78")
        self.fLabel_8 = QtWidgets.QLabel(parent=self.FitGroupConst_6)
        self.fLabel_8.setMinimumSize(QtCore.QSize(50, 30))
        self.fLabel_8.setMaximumSize(QtCore.QSize(50, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.fLabel_8.setFont(font)
        self.fLabel_8.setObjectName("fLabel_8")
        self.horizontalLayout_78.addWidget(self.fLabel_8)
        self.fTextEdit_05c = QtWidgets.QLineEdit(parent=self.FitGroupConst_6)
        self.fTextEdit_05c.setMinimumSize(QtCore.QSize(200, 23))
        self.fTextEdit_05c.setMaximumSize(QtCore.QSize(200, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.fTextEdit_05c.setFont(font)
        self.fTextEdit_05c.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhFormattedNumbersOnly)
        self.fTextEdit_05c.setText("")
        self.fTextEdit_05c.setObjectName("fTextEdit_05c")
        self.horizontalLayout_78.addWidget(self.fTextEdit_05c)
        self.gridLayout_15.addLayout(self.horizontalLayout_78, 6, 0, 1, 1)
        self.horizontalLayout_79 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_79.setObjectName("horizontalLayout_79")
        self.fdLabel_8 = QtWidgets.QLabel(parent=self.FitGroupConst_6)
        self.fdLabel_8.setMinimumSize(QtCore.QSize(50, 30))
        self.fdLabel_8.setMaximumSize(QtCore.QSize(50, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.fdLabel_8.setFont(font)
        self.fdLabel_8.setObjectName("fdLabel_8")
        self.horizontalLayout_79.addWidget(self.fdLabel_8)
        self.fdTextEdit_05c = QtWidgets.QLineEdit(parent=self.FitGroupConst_6)
        self.fdTextEdit_05c.setMinimumSize(QtCore.QSize(200, 23))
        self.fdTextEdit_05c.setMaximumSize(QtCore.QSize(200, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.fdTextEdit_05c.setFont(font)
        self.fdTextEdit_05c.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhFormattedNumbersOnly)
        self.fdTextEdit_05c.setText("")
        self.fdTextEdit_05c.setObjectName("fdTextEdit_05c")
        self.horizontalLayout_79.addWidget(self.fdTextEdit_05c)
        self.gridLayout_15.addLayout(self.horizontalLayout_79, 7, 0, 1, 1)
        self.horizontalLayout_161.addWidget(self.FitGroupConst_6)
        self.verticalLayout_36.addLayout(self.horizontalLayout_161)
        self.horizontalLayout_162 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_162.setObjectName("horizontalLayout_162")
        self.cyc_long_life = QtWidgets.QCheckBox(parent=self.tab_3)
        self.cyc_long_life.setMinimumSize(QtCore.QSize(200, 30))
        self.cyc_long_life.setMaximumSize(QtCore.QSize(200, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.cyc_long_life.setFont(font)
        self.cyc_long_life.setChecked(True)
        self.cyc_long_life.setObjectName("cyc_long_life")
        self.horizontalLayout_162.addWidget(self.cyc_long_life)
        self.simul_long_life = QtWidgets.QCheckBox(parent=self.tab_3)
        self.simul_long_life.setMinimumSize(QtCore.QSize(200, 30))
        self.simul_long_life.setMaximumSize(QtCore.QSize(200, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.simul_long_life.setFont(font)
        self.simul_long_life.setChecked(False)
        self.simul_long_life.setObjectName("simul_long_life")
        self.horizontalLayout_162.addWidget(self.simul_long_life)
        self.verticalLayout_36.addLayout(self.horizontalLayout_162)
        self.horizontalLayout_163 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_163.setObjectName("horizontalLayout_163")
        self.cycxlabel_33 = QtWidgets.QLabel(parent=self.tab_3)
        self.cycxlabel_33.setMinimumSize(QtCore.QSize(200, 25))
        self.cycxlabel_33.setMaximumSize(QtCore.QSize(200, 25))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.cycxlabel_33.setFont(font)
        self.cycxlabel_33.setObjectName("cycxlabel_33")
        self.horizontalLayout_163.addWidget(self.cycxlabel_33)
        self.simul_y_max = QtWidgets.QLineEdit(parent=self.tab_3)
        self.simul_y_max.setMinimumSize(QtCore.QSize(300, 25))
        self.simul_y_max.setMaximumSize(QtCore.QSize(300, 25))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.simul_y_max.setFont(font)
        self.simul_y_max.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhFormattedNumbersOnly)
        self.simul_y_max.setObjectName("simul_y_max")
        self.horizontalLayout_163.addWidget(self.simul_y_max)
        self.verticalLayout_36.addLayout(self.horizontalLayout_163)
        self.horizontalLayout_164 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_164.setObjectName("horizontalLayout_164")
        self.cycxlabel_17 = QtWidgets.QLabel(parent=self.tab_3)
        self.cycxlabel_17.setMinimumSize(QtCore.QSize(200, 25))
        self.cycxlabel_17.setMaximumSize(QtCore.QSize(200, 25))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.cycxlabel_17.setFont(font)
        self.cycxlabel_17.setObjectName("cycxlabel_17")
        self.horizontalLayout_164.addWidget(self.cycxlabel_17)
        self.simul_y_min = QtWidgets.QLineEdit(parent=self.tab_3)
        self.simul_y_min.setMinimumSize(QtCore.QSize(300, 25))
        self.simul_y_min.setMaximumSize(QtCore.QSize(300, 25))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.simul_y_min.setFont(font)
        self.simul_y_min.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhFormattedNumbersOnly)
        self.simul_y_min.setObjectName("simul_y_min")
        self.horizontalLayout_164.addWidget(self.simul_y_min)
        self.verticalLayout_36.addLayout(self.horizontalLayout_164)
        self.horizontalLayout_165 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_165.setObjectName("horizontalLayout_165")
        self.cycxlabel_34 = QtWidgets.QLabel(parent=self.tab_3)
        self.cycxlabel_34.setMinimumSize(QtCore.QSize(200, 25))
        self.cycxlabel_34.setMaximumSize(QtCore.QSize(200, 25))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.cycxlabel_34.setFont(font)
        self.cycxlabel_34.setObjectName("cycxlabel_34")
        self.horizontalLayout_165.addWidget(self.cycxlabel_34)
        self.simul_x_max = QtWidgets.QLineEdit(parent=self.tab_3)
        self.simul_x_max.setMinimumSize(QtCore.QSize(300, 25))
        self.simul_x_max.setMaximumSize(QtCore.QSize(300, 25))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.simul_x_max.setFont(font)
        self.simul_x_max.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhDigitsOnly)
        self.simul_x_max.setObjectName("simul_x_max")
        self.horizontalLayout_165.addWidget(self.simul_x_max)
        self.verticalLayout_36.addLayout(self.horizontalLayout_165)
        self.horizontalLayout_166 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_166.setObjectName("horizontalLayout_166")
        self.load_cycparameter = QtWidgets.QPushButton(parent=self.tab_3)
        self.load_cycparameter.setMinimumSize(QtCore.QSize(315, 80))
        self.load_cycparameter.setMaximumSize(QtCore.QSize(315, 80))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.load_cycparameter.setFont(font)
        self.load_cycparameter.setObjectName("load_cycparameter")
        self.horizontalLayout_166.addWidget(self.load_cycparameter)
        self.AppCycleTabReset = QtWidgets.QPushButton(parent=self.tab_3)
        self.AppCycleTabReset.setMinimumSize(QtCore.QSize(315, 80))
        self.AppCycleTabReset.setMaximumSize(QtCore.QSize(315, 80))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(True)
        font.setItalic(False)
        font.setUnderline(True)
        font.setWeight(75)
        self.AppCycleTabReset.setFont(font)
        self.AppCycleTabReset.setObjectName("AppCycleTabReset")
        self.horizontalLayout_166.addWidget(self.AppCycleTabReset)
        self.verticalLayout_36.addLayout(self.horizontalLayout_166)
        self.horizontalLayout_167 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_167.setObjectName("horizontalLayout_167")
        self.pathappcycestimation = QtWidgets.QPushButton(parent=self.tab_3)
        self.pathappcycestimation.setMinimumSize(QtCore.QSize(315, 80))
        self.pathappcycestimation.setMaximumSize(QtCore.QSize(315, 80))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.pathappcycestimation.setFont(font)
        self.pathappcycestimation.setObjectName("pathappcycestimation")
        self.horizontalLayout_167.addWidget(self.pathappcycestimation)
        self.folderappcycestimation = QtWidgets.QPushButton(parent=self.tab_3)
        self.folderappcycestimation.setMinimumSize(QtCore.QSize(315, 80))
        self.folderappcycestimation.setMaximumSize(QtCore.QSize(315, 80))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.folderappcycestimation.setFont(font)
        self.folderappcycestimation.setObjectName("folderappcycestimation")
        self.horizontalLayout_167.addWidget(self.folderappcycestimation)
        self.verticalLayout_36.addLayout(self.horizontalLayout_167)
        self.horizontalLayout_168.addLayout(self.verticalLayout_36)
        self.cycle_simul_tab = QtWidgets.QTabWidget(parent=self.tab_3)
        self.cycle_simul_tab.setMinimumSize(QtCore.QSize(1200, 830))
        self.cycle_simul_tab.setMaximumSize(QtCore.QSize(1200, 830))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.cycle_simul_tab.setFont(font)
        self.cycle_simul_tab.setObjectName("cycle_simul_tab")
        self.horizontalLayout_168.addWidget(self.cycle_simul_tab)
        self.horizontalLayout_169.addLayout(self.horizontalLayout_168)
        self.tabWidget.addTab(self.tab_3, "")
        self.FitTab = QtWidgets.QWidget()
        self.FitTab.setObjectName("FitTab")
        self.horizontalLayout_171 = QtWidgets.QHBoxLayout(self.FitTab)
        self.horizontalLayout_171.setObjectName("horizontalLayout_171")
        self.horizontalLayout_170 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_170.setObjectName("horizontalLayout_170")
        self.verticalLayout_37 = QtWidgets.QVBoxLayout()
        self.verticalLayout_37.setObjectName("verticalLayout_37")
        self.horizontalLayout_16 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_16.setObjectName("horizontalLayout_16")
        self.selectcycle = QtWidgets.QGroupBox(parent=self.FitTab)
        self.selectcycle.setMinimumSize(QtCore.QSize(125, 80))
        self.selectcycle.setMaximumSize(QtCore.QSize(125, 80))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.selectcycle.setFont(font)
        self.selectcycle.setObjectName("selectcycle")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.selectcycle)
        self.verticalLayout.setObjectName("verticalLayout")
        self.chkcapacity = QtWidgets.QRadioButton(parent=self.selectcycle)
        self.chkcapacity.setMinimumSize(QtCore.QSize(100, 21))
        self.chkcapacity.setMaximumSize(QtCore.QSize(100, 21))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.chkcapacity.setFont(font)
        self.chkcapacity.setChecked(True)
        self.chkcapacity.setObjectName("chkcapacity")
        self.verticalLayout.addWidget(self.chkcapacity)
        self.chkdcir = QtWidgets.QRadioButton(parent=self.selectcycle)
        self.chkdcir.setMinimumSize(QtCore.QSize(100, 21))
        self.chkdcir.setMaximumSize(QtCore.QSize(100, 21))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.chkdcir.setFont(font)
        self.chkdcir.setObjectName("chkdcir")
        self.verticalLayout.addWidget(self.chkdcir)
        self.horizontalLayout_16.addWidget(self.selectcycle)
        self.selectcapacity = QtWidgets.QGroupBox(parent=self.FitTab)
        self.selectcapacity.setMinimumSize(QtCore.QSize(125, 80))
        self.selectcapacity.setMaximumSize(QtCore.QSize(125, 80))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.selectcapacity.setFont(font)
        self.selectcapacity.setObjectName("selectcapacity")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.selectcapacity)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.chkcycle = QtWidgets.QRadioButton(parent=self.selectcapacity)
        self.chkcycle.setMinimumSize(QtCore.QSize(100, 21))
        self.chkcycle.setMaximumSize(QtCore.QSize(100, 21))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.chkcycle.setFont(font)
        self.chkcycle.setChecked(True)
        self.chkcycle.setObjectName("chkcycle")
        self.verticalLayout_2.addWidget(self.chkcycle)
        self.chkstorage = QtWidgets.QRadioButton(parent=self.selectcapacity)
        self.chkstorage.setMinimumSize(QtCore.QSize(100, 21))
        self.chkstorage.setMaximumSize(QtCore.QSize(100, 21))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.chkstorage.setFont(font)
        self.chkstorage.setObjectName("chkstorage")
        self.verticalLayout_2.addWidget(self.chkstorage)
        self.horizontalLayout_16.addWidget(self.selectcapacity)
        self.selectlonglifecyc = QtWidgets.QGroupBox(parent=self.FitTab)
        self.selectlonglifecyc.setMinimumSize(QtCore.QSize(124, 80))
        self.selectlonglifecyc.setMaximumSize(QtCore.QSize(124, 80))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.selectlonglifecyc.setFont(font)
        self.selectlonglifecyc.setObjectName("selectlonglifecyc")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.selectlonglifecyc)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.nolonglife = QtWidgets.QRadioButton(parent=self.selectlonglifecyc)
        self.nolonglife.setMinimumSize(QtCore.QSize(100, 21))
        self.nolonglife.setMaximumSize(QtCore.QSize(100, 21))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.nolonglife.setFont(font)
        self.nolonglife.setChecked(True)
        self.nolonglife.setObjectName("nolonglife")
        self.verticalLayout_3.addWidget(self.nolonglife)
        self.hhp_longlife = QtWidgets.QRadioButton(parent=self.selectlonglifecyc)
        self.hhp_longlife.setMinimumSize(QtCore.QSize(100, 21))
        self.hhp_longlife.setMaximumSize(QtCore.QSize(100, 21))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.hhp_longlife.setFont(font)
        self.hhp_longlife.setObjectName("hhp_longlife")
        self.verticalLayout_3.addWidget(self.hhp_longlife)
        self.horizontalLayout_16.addWidget(self.selectlonglifecyc)
        self.SimulTabResetConfirm = QtWidgets.QPushButton(parent=self.FitTab)
        self.SimulTabResetConfirm.setMinimumSize(QtCore.QSize(125, 80))
        self.SimulTabResetConfirm.setMaximumSize(QtCore.QSize(125, 80))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(True)
        font.setUnderline(True)
        font.setWeight(75)
        font.setStrikeOut(False)
        font.setKerning(True)
        self.SimulTabResetConfirm.setFont(font)
        self.SimulTabResetConfirm.setObjectName("SimulTabResetConfirm")
        self.horizontalLayout_16.addWidget(self.SimulTabResetConfirm)
        self.SimulConfirm = QtWidgets.QPushButton(parent=self.FitTab)
        self.SimulConfirm.setMinimumSize(QtCore.QSize(125, 80))
        self.SimulConfirm.setMaximumSize(QtCore.QSize(125, 80))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.SimulConfirm.setFont(font)
        self.SimulConfirm.setObjectName("SimulConfirm")
        self.horizontalLayout_16.addWidget(self.SimulConfirm)
        self.verticalLayout_37.addLayout(self.horizontalLayout_16)
        self.horizontalLayout_80 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_80.setObjectName("horizontalLayout_80")
        self.label = QtWidgets.QLabel(parent=self.FitTab)
        self.label.setMinimumSize(QtCore.QSize(60, 20))
        self.label.setMaximumSize(QtCore.QSize(60, 20))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.horizontalLayout_80.addWidget(self.label)
        self.txt_longcycleno = QtWidgets.QLineEdit(parent=self.FitTab)
        self.txt_longcycleno.setMinimumSize(QtCore.QSize(135, 20))
        self.txt_longcycleno.setMaximumSize(QtCore.QSize(135, 20))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.txt_longcycleno.setFont(font)
        self.txt_longcycleno.setObjectName("txt_longcycleno")
        self.horizontalLayout_80.addWidget(self.txt_longcycleno)
        self.txt_longcyclevol = QtWidgets.QLineEdit(parent=self.FitTab)
        self.txt_longcyclevol.setMinimumSize(QtCore.QSize(135, 20))
        self.txt_longcyclevol.setMaximumSize(QtCore.QSize(135, 20))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.txt_longcyclevol.setFont(font)
        self.txt_longcyclevol.setObjectName("txt_longcyclevol")
        self.horizontalLayout_80.addWidget(self.txt_longcyclevol)
        self.txt_relcap = QtWidgets.QLineEdit(parent=self.FitTab)
        self.txt_relcap.setMinimumSize(QtCore.QSize(135, 20))
        self.txt_relcap.setMaximumSize(QtCore.QSize(135, 20))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.txt_relcap.setFont(font)
        self.txt_relcap.setObjectName("txt_relcap")
        self.horizontalLayout_80.addWidget(self.txt_relcap)
        self.verticalLayout_37.addLayout(self.horizontalLayout_80)
        self.horizontalLayout_62 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_62.setObjectName("horizontalLayout_62")
        self.label_8 = QtWidgets.QLabel(parent=self.FitTab)
        self.label_8.setMinimumSize(QtCore.QSize(200, 20))
        self.label_8.setMaximumSize(QtCore.QSize(200, 20))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.label_8.setFont(font)
        self.label_8.setObjectName("label_8")
        self.horizontalLayout_62.addWidget(self.label_8)
        self.txt_storageratio = QtWidgets.QLineEdit(parent=self.FitTab)
        self.txt_storageratio.setMinimumSize(QtCore.QSize(30, 20))
        self.txt_storageratio.setMaximumSize(QtCore.QSize(30, 20))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.txt_storageratio.setFont(font)
        self.txt_storageratio.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.txt_storageratio.setObjectName("txt_storageratio")
        self.horizontalLayout_62.addWidget(self.txt_storageratio)
        self.label_16 = QtWidgets.QLabel(parent=self.FitTab)
        self.label_16.setMinimumSize(QtCore.QSize(200, 20))
        self.label_16.setMaximumSize(QtCore.QSize(200, 20))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.label_16.setFont(font)
        self.label_16.setObjectName("label_16")
        self.horizontalLayout_62.addWidget(self.label_16)
        self.txt_storageratio2 = QtWidgets.QLineEdit(parent=self.FitTab)
        self.txt_storageratio2.setMinimumSize(QtCore.QSize(30, 20))
        self.txt_storageratio2.setMaximumSize(QtCore.QSize(30, 20))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.txt_storageratio2.setFont(font)
        self.txt_storageratio2.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.txt_storageratio2.setObjectName("txt_storageratio2")
        self.horizontalLayout_62.addWidget(self.txt_storageratio2)
        self.verticalLayout_37.addLayout(self.horizontalLayout_62)
        self.horizontalLayout_21 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_21.setObjectName("horizontalLayout_21")
        self.cycxlabel_53 = QtWidgets.QLabel(parent=self.FitTab)
        self.cycxlabel_53.setMinimumSize(QtCore.QSize(140, 30))
        self.cycxlabel_53.setMaximumSize(QtCore.QSize(140, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.cycxlabel_53.setFont(font)
        self.cycxlabel_53.setObjectName("cycxlabel_53")
        self.horizontalLayout_21.addWidget(self.cycxlabel_53)
        self.chk_cell_cycle = QtWidgets.QCheckBox(parent=self.FitTab)
        self.chk_cell_cycle.setMinimumSize(QtCore.QSize(163, 18))
        self.chk_cell_cycle.setMaximumSize(QtCore.QSize(163, 18))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.chk_cell_cycle.setFont(font)
        self.chk_cell_cycle.setChecked(True)
        self.chk_cell_cycle.setObjectName("chk_cell_cycle")
        self.horizontalLayout_21.addWidget(self.chk_cell_cycle)
        self.chk_set_cycle = QtWidgets.QCheckBox(parent=self.FitTab)
        self.chk_set_cycle.setMinimumSize(QtCore.QSize(164, 18))
        self.chk_set_cycle.setMaximumSize(QtCore.QSize(164, 18))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.chk_set_cycle.setFont(font)
        self.chk_set_cycle.setChecked(True)
        self.chk_set_cycle.setObjectName("chk_set_cycle")
        self.horizontalLayout_21.addWidget(self.chk_set_cycle)
        self.chk_detail_cycle = QtWidgets.QCheckBox(parent=self.FitTab)
        self.chk_detail_cycle.setMinimumSize(QtCore.QSize(163, 18))
        self.chk_detail_cycle.setMaximumSize(QtCore.QSize(163, 18))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.chk_detail_cycle.setFont(font)
        self.chk_detail_cycle.setChecked(True)
        self.chk_detail_cycle.setObjectName("chk_detail_cycle")
        self.horizontalLayout_21.addWidget(self.chk_detail_cycle)
        self.verticalLayout_37.addLayout(self.horizontalLayout_21)
        self.capparameterload_path = QtWidgets.QLineEdit(parent=self.FitTab)
        self.capparameterload_path.setMinimumSize(QtCore.QSize(642, 20))
        self.capparameterload_path.setMaximumSize(QtCore.QSize(642, 20))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.capparameterload_path.setFont(font)
        self.capparameterload_path.setText("")
        self.capparameterload_path.setObjectName("capparameterload_path")
        self.verticalLayout_37.addWidget(self.capparameterload_path)
        self.horizontalLayout_61 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_61.setObjectName("horizontalLayout_61")
        self.FitGroupConst = QtWidgets.QGroupBox(parent=self.FitTab)
        self.FitGroupConst.setMinimumSize(QtCore.QSize(150, 334))
        self.FitGroupConst.setMaximumSize(QtCore.QSize(150, 334))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.FitGroupConst.setFont(font)
        self.FitGroupConst.setObjectName("FitGroupConst")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.FitGroupConst)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.horizontalLayout_26 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_26.setObjectName("horizontalLayout_26")
        self.aLabel = QtWidgets.QLabel(parent=self.FitGroupConst)
        self.aLabel.setMinimumSize(QtCore.QSize(50, 30))
        self.aLabel.setMaximumSize(QtCore.QSize(50, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.aLabel.setFont(font)
        self.aLabel.setObjectName("aLabel")
        self.horizontalLayout_26.addWidget(self.aLabel)
        self.aTextEdit = QtWidgets.QLineEdit(parent=self.FitGroupConst)
        self.aTextEdit.setMinimumSize(QtCore.QSize(72, 23))
        self.aTextEdit.setMaximumSize(QtCore.QSize(72, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.aTextEdit.setFont(font)
        self.aTextEdit.setText("")
        self.aTextEdit.setObjectName("aTextEdit")
        self.horizontalLayout_26.addWidget(self.aTextEdit)
        self.gridLayout_4.addLayout(self.horizontalLayout_26, 0, 0, 1, 1)
        self.horizontalLayout_27 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_27.setObjectName("horizontalLayout_27")
        self.bLabel = QtWidgets.QLabel(parent=self.FitGroupConst)
        self.bLabel.setMinimumSize(QtCore.QSize(50, 30))
        self.bLabel.setMaximumSize(QtCore.QSize(50, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.bLabel.setFont(font)
        self.bLabel.setObjectName("bLabel")
        self.horizontalLayout_27.addWidget(self.bLabel)
        self.bTextEdit = QtWidgets.QLineEdit(parent=self.FitGroupConst)
        self.bTextEdit.setMinimumSize(QtCore.QSize(72, 23))
        self.bTextEdit.setMaximumSize(QtCore.QSize(72, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.bTextEdit.setFont(font)
        self.bTextEdit.setText("")
        self.bTextEdit.setObjectName("bTextEdit")
        self.horizontalLayout_27.addWidget(self.bTextEdit)
        self.gridLayout_4.addLayout(self.horizontalLayout_27, 1, 0, 1, 1)
        self.horizontalLayout_28 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_28.setObjectName("horizontalLayout_28")
        self.b1Label = QtWidgets.QLabel(parent=self.FitGroupConst)
        self.b1Label.setMinimumSize(QtCore.QSize(50, 30))
        self.b1Label.setMaximumSize(QtCore.QSize(50, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.b1Label.setFont(font)
        self.b1Label.setObjectName("b1Label")
        self.horizontalLayout_28.addWidget(self.b1Label)
        self.b1TextEdit = QtWidgets.QLineEdit(parent=self.FitGroupConst)
        self.b1TextEdit.setMinimumSize(QtCore.QSize(72, 23))
        self.b1TextEdit.setMaximumSize(QtCore.QSize(72, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.b1TextEdit.setFont(font)
        self.b1TextEdit.setText("")
        self.b1TextEdit.setObjectName("b1TextEdit")
        self.horizontalLayout_28.addWidget(self.b1TextEdit)
        self.gridLayout_4.addLayout(self.horizontalLayout_28, 2, 0, 1, 1)
        self.horizontalLayout_29 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_29.setObjectName("horizontalLayout_29")
        self.cLabel = QtWidgets.QLabel(parent=self.FitGroupConst)
        self.cLabel.setMinimumSize(QtCore.QSize(50, 30))
        self.cLabel.setMaximumSize(QtCore.QSize(50, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.cLabel.setFont(font)
        self.cLabel.setObjectName("cLabel")
        self.horizontalLayout_29.addWidget(self.cLabel)
        self.cTextEdit = QtWidgets.QLineEdit(parent=self.FitGroupConst)
        self.cTextEdit.setMinimumSize(QtCore.QSize(72, 23))
        self.cTextEdit.setMaximumSize(QtCore.QSize(72, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.cTextEdit.setFont(font)
        self.cTextEdit.setText("")
        self.cTextEdit.setObjectName("cTextEdit")
        self.horizontalLayout_29.addWidget(self.cTextEdit)
        self.gridLayout_4.addLayout(self.horizontalLayout_29, 3, 0, 1, 1)
        self.horizontalLayout_30 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_30.setObjectName("horizontalLayout_30")
        self.dLabel = QtWidgets.QLabel(parent=self.FitGroupConst)
        self.dLabel.setMinimumSize(QtCore.QSize(50, 29))
        self.dLabel.setMaximumSize(QtCore.QSize(50, 29))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.dLabel.setFont(font)
        self.dLabel.setObjectName("dLabel")
        self.horizontalLayout_30.addWidget(self.dLabel)
        self.dTextEdit = QtWidgets.QLineEdit(parent=self.FitGroupConst)
        self.dTextEdit.setMinimumSize(QtCore.QSize(72, 23))
        self.dTextEdit.setMaximumSize(QtCore.QSize(72, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.dTextEdit.setFont(font)
        self.dTextEdit.setText("")
        self.dTextEdit.setObjectName("dTextEdit")
        self.horizontalLayout_30.addWidget(self.dTextEdit)
        self.gridLayout_4.addLayout(self.horizontalLayout_30, 4, 0, 1, 1)
        self.horizontalLayout_31 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_31.setObjectName("horizontalLayout_31")
        self.eLabel = QtWidgets.QLabel(parent=self.FitGroupConst)
        self.eLabel.setMinimumSize(QtCore.QSize(50, 30))
        self.eLabel.setMaximumSize(QtCore.QSize(50, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.eLabel.setFont(font)
        self.eLabel.setObjectName("eLabel")
        self.horizontalLayout_31.addWidget(self.eLabel)
        self.eTextEdit = QtWidgets.QLineEdit(parent=self.FitGroupConst)
        self.eTextEdit.setMinimumSize(QtCore.QSize(72, 23))
        self.eTextEdit.setMaximumSize(QtCore.QSize(72, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.eTextEdit.setFont(font)
        self.eTextEdit.setText("")
        self.eTextEdit.setObjectName("eTextEdit")
        self.horizontalLayout_31.addWidget(self.eTextEdit)
        self.gridLayout_4.addLayout(self.horizontalLayout_31, 5, 0, 1, 1)
        self.horizontalLayout_34 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_34.setObjectName("horizontalLayout_34")
        self.fLabel = QtWidgets.QLabel(parent=self.FitGroupConst)
        self.fLabel.setMinimumSize(QtCore.QSize(50, 30))
        self.fLabel.setMaximumSize(QtCore.QSize(50, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.fLabel.setFont(font)
        self.fLabel.setObjectName("fLabel")
        self.horizontalLayout_34.addWidget(self.fLabel)
        self.fTextEdit = QtWidgets.QLineEdit(parent=self.FitGroupConst)
        self.fTextEdit.setMinimumSize(QtCore.QSize(72, 23))
        self.fTextEdit.setMaximumSize(QtCore.QSize(72, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.fTextEdit.setFont(font)
        self.fTextEdit.setText("")
        self.fTextEdit.setObjectName("fTextEdit")
        self.horizontalLayout_34.addWidget(self.fTextEdit)
        self.gridLayout_4.addLayout(self.horizontalLayout_34, 6, 0, 1, 1)
        self.horizontalLayout_36 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_36.setObjectName("horizontalLayout_36")
        self.fdLabel = QtWidgets.QLabel(parent=self.FitGroupConst)
        self.fdLabel.setMinimumSize(QtCore.QSize(50, 30))
        self.fdLabel.setMaximumSize(QtCore.QSize(50, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.fdLabel.setFont(font)
        self.fdLabel.setObjectName("fdLabel")
        self.horizontalLayout_36.addWidget(self.fdLabel)
        self.fdTextEdit = QtWidgets.QLineEdit(parent=self.FitGroupConst)
        self.fdTextEdit.setMinimumSize(QtCore.QSize(72, 23))
        self.fdTextEdit.setMaximumSize(QtCore.QSize(72, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.fdTextEdit.setFont(font)
        self.fdTextEdit.setText("")
        self.fdTextEdit.setObjectName("fdTextEdit")
        self.horizontalLayout_36.addWidget(self.fdTextEdit)
        self.gridLayout_4.addLayout(self.horizontalLayout_36, 7, 0, 1, 1)
        self.horizontalLayout_61.addWidget(self.FitGroupConst)
        self.FitGroupConst_3 = QtWidgets.QGroupBox(parent=self.FitTab)
        self.FitGroupConst_3.setMinimumSize(QtCore.QSize(150, 334))
        self.FitGroupConst_3.setMaximumSize(QtCore.QSize(150, 334))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.FitGroupConst_3.setFont(font)
        self.FitGroupConst_3.setObjectName("FitGroupConst_3")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.FitGroupConst_3)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.horizontalLayout_37 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_37.setObjectName("horizontalLayout_37")
        self.aLabel_5 = QtWidgets.QLabel(parent=self.FitGroupConst_3)
        self.aLabel_5.setMinimumSize(QtCore.QSize(50, 30))
        self.aLabel_5.setMaximumSize(QtCore.QSize(50, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.aLabel_5.setFont(font)
        self.aLabel_5.setObjectName("aLabel_5")
        self.horizontalLayout_37.addWidget(self.aLabel_5)
        self.aTextEdit_3 = QtWidgets.QLineEdit(parent=self.FitGroupConst_3)
        self.aTextEdit_3.setMinimumSize(QtCore.QSize(72, 23))
        self.aTextEdit_3.setMaximumSize(QtCore.QSize(72, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.aTextEdit_3.setFont(font)
        self.aTextEdit_3.setText("")
        self.aTextEdit_3.setObjectName("aTextEdit_3")
        self.horizontalLayout_37.addWidget(self.aTextEdit_3)
        self.gridLayout_5.addLayout(self.horizontalLayout_37, 0, 0, 1, 1)
        self.horizontalLayout_38 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_38.setObjectName("horizontalLayout_38")
        self.bLabel_5 = QtWidgets.QLabel(parent=self.FitGroupConst_3)
        self.bLabel_5.setMinimumSize(QtCore.QSize(50, 30))
        self.bLabel_5.setMaximumSize(QtCore.QSize(50, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.bLabel_5.setFont(font)
        self.bLabel_5.setObjectName("bLabel_5")
        self.horizontalLayout_38.addWidget(self.bLabel_5)
        self.bTextEdit_3 = QtWidgets.QLineEdit(parent=self.FitGroupConst_3)
        self.bTextEdit_3.setMinimumSize(QtCore.QSize(72, 23))
        self.bTextEdit_3.setMaximumSize(QtCore.QSize(72, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.bTextEdit_3.setFont(font)
        self.bTextEdit_3.setText("")
        self.bTextEdit_3.setObjectName("bTextEdit_3")
        self.horizontalLayout_38.addWidget(self.bTextEdit_3)
        self.gridLayout_5.addLayout(self.horizontalLayout_38, 1, 0, 1, 1)
        self.horizontalLayout_39 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_39.setObjectName("horizontalLayout_39")
        self.b1Label_5 = QtWidgets.QLabel(parent=self.FitGroupConst_3)
        self.b1Label_5.setMinimumSize(QtCore.QSize(50, 30))
        self.b1Label_5.setMaximumSize(QtCore.QSize(50, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.b1Label_5.setFont(font)
        self.b1Label_5.setObjectName("b1Label_5")
        self.horizontalLayout_39.addWidget(self.b1Label_5)
        self.b1TextEdit_3 = QtWidgets.QLineEdit(parent=self.FitGroupConst_3)
        self.b1TextEdit_3.setMinimumSize(QtCore.QSize(72, 23))
        self.b1TextEdit_3.setMaximumSize(QtCore.QSize(72, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.b1TextEdit_3.setFont(font)
        self.b1TextEdit_3.setText("")
        self.b1TextEdit_3.setObjectName("b1TextEdit_3")
        self.horizontalLayout_39.addWidget(self.b1TextEdit_3)
        self.gridLayout_5.addLayout(self.horizontalLayout_39, 2, 0, 1, 1)
        self.horizontalLayout_40 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_40.setObjectName("horizontalLayout_40")
        self.cLabel_5 = QtWidgets.QLabel(parent=self.FitGroupConst_3)
        self.cLabel_5.setMinimumSize(QtCore.QSize(50, 30))
        self.cLabel_5.setMaximumSize(QtCore.QSize(50, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.cLabel_5.setFont(font)
        self.cLabel_5.setObjectName("cLabel_5")
        self.horizontalLayout_40.addWidget(self.cLabel_5)
        self.cTextEdit_3 = QtWidgets.QLineEdit(parent=self.FitGroupConst_3)
        self.cTextEdit_3.setMinimumSize(QtCore.QSize(72, 23))
        self.cTextEdit_3.setMaximumSize(QtCore.QSize(72, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.cTextEdit_3.setFont(font)
        self.cTextEdit_3.setText("")
        self.cTextEdit_3.setObjectName("cTextEdit_3")
        self.horizontalLayout_40.addWidget(self.cTextEdit_3)
        self.gridLayout_5.addLayout(self.horizontalLayout_40, 3, 0, 1, 1)
        self.horizontalLayout_41 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_41.setObjectName("horizontalLayout_41")
        self.dLabel_5 = QtWidgets.QLabel(parent=self.FitGroupConst_3)
        self.dLabel_5.setMinimumSize(QtCore.QSize(50, 29))
        self.dLabel_5.setMaximumSize(QtCore.QSize(50, 29))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.dLabel_5.setFont(font)
        self.dLabel_5.setObjectName("dLabel_5")
        self.horizontalLayout_41.addWidget(self.dLabel_5)
        self.dTextEdit_3 = QtWidgets.QLineEdit(parent=self.FitGroupConst_3)
        self.dTextEdit_3.setMinimumSize(QtCore.QSize(72, 23))
        self.dTextEdit_3.setMaximumSize(QtCore.QSize(72, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.dTextEdit_3.setFont(font)
        self.dTextEdit_3.setText("")
        self.dTextEdit_3.setObjectName("dTextEdit_3")
        self.horizontalLayout_41.addWidget(self.dTextEdit_3)
        self.gridLayout_5.addLayout(self.horizontalLayout_41, 4, 0, 1, 1)
        self.horizontalLayout_42 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_42.setObjectName("horizontalLayout_42")
        self.eLabel_5 = QtWidgets.QLabel(parent=self.FitGroupConst_3)
        self.eLabel_5.setMinimumSize(QtCore.QSize(50, 30))
        self.eLabel_5.setMaximumSize(QtCore.QSize(50, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.eLabel_5.setFont(font)
        self.eLabel_5.setObjectName("eLabel_5")
        self.horizontalLayout_42.addWidget(self.eLabel_5)
        self.eTextEdit_3 = QtWidgets.QLineEdit(parent=self.FitGroupConst_3)
        self.eTextEdit_3.setMinimumSize(QtCore.QSize(72, 23))
        self.eTextEdit_3.setMaximumSize(QtCore.QSize(72, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.eTextEdit_3.setFont(font)
        self.eTextEdit_3.setText("")
        self.eTextEdit_3.setObjectName("eTextEdit_3")
        self.horizontalLayout_42.addWidget(self.eTextEdit_3)
        self.gridLayout_5.addLayout(self.horizontalLayout_42, 5, 0, 1, 1)
        self.horizontalLayout_43 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_43.setObjectName("horizontalLayout_43")
        self.fLabel_5 = QtWidgets.QLabel(parent=self.FitGroupConst_3)
        self.fLabel_5.setMinimumSize(QtCore.QSize(50, 30))
        self.fLabel_5.setMaximumSize(QtCore.QSize(50, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.fLabel_5.setFont(font)
        self.fLabel_5.setObjectName("fLabel_5")
        self.horizontalLayout_43.addWidget(self.fLabel_5)
        self.fTextEdit_3 = QtWidgets.QLineEdit(parent=self.FitGroupConst_3)
        self.fTextEdit_3.setMinimumSize(QtCore.QSize(72, 23))
        self.fTextEdit_3.setMaximumSize(QtCore.QSize(72, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.fTextEdit_3.setFont(font)
        self.fTextEdit_3.setText("")
        self.fTextEdit_3.setObjectName("fTextEdit_3")
        self.horizontalLayout_43.addWidget(self.fTextEdit_3)
        self.gridLayout_5.addLayout(self.horizontalLayout_43, 6, 0, 1, 1)
        self.horizontalLayout_44 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_44.setObjectName("horizontalLayout_44")
        self.fdLabel_5 = QtWidgets.QLabel(parent=self.FitGroupConst_3)
        self.fdLabel_5.setMinimumSize(QtCore.QSize(50, 30))
        self.fdLabel_5.setMaximumSize(QtCore.QSize(50, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.fdLabel_5.setFont(font)
        self.fdLabel_5.setObjectName("fdLabel_5")
        self.horizontalLayout_44.addWidget(self.fdLabel_5)
        self.fdTextEdit_3 = QtWidgets.QLineEdit(parent=self.FitGroupConst_3)
        self.fdTextEdit_3.setMinimumSize(QtCore.QSize(72, 23))
        self.fdTextEdit_3.setMaximumSize(QtCore.QSize(72, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.fdTextEdit_3.setFont(font)
        self.fdTextEdit_3.setText("")
        self.fdTextEdit_3.setObjectName("fdTextEdit_3")
        self.horizontalLayout_44.addWidget(self.fdTextEdit_3)
        self.gridLayout_5.addLayout(self.horizontalLayout_44, 7, 0, 1, 1)
        self.horizontalLayout_61.addWidget(self.FitGroupConst_3)
        self.FitGroupConst_2 = QtWidgets.QGroupBox(parent=self.FitTab)
        self.FitGroupConst_2.setMinimumSize(QtCore.QSize(150, 334))
        self.FitGroupConst_2.setMaximumSize(QtCore.QSize(150, 334))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.FitGroupConst_2.setFont(font)
        self.FitGroupConst_2.setObjectName("FitGroupConst_2")
        self.gridLayout_6 = QtWidgets.QGridLayout(self.FitGroupConst_2)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.horizontalLayout_45 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_45.setObjectName("horizontalLayout_45")
        self.aLabel_6 = QtWidgets.QLabel(parent=self.FitGroupConst_2)
        self.aLabel_6.setMinimumSize(QtCore.QSize(50, 30))
        self.aLabel_6.setMaximumSize(QtCore.QSize(50, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.aLabel_6.setFont(font)
        self.aLabel_6.setObjectName("aLabel_6")
        self.horizontalLayout_45.addWidget(self.aLabel_6)
        self.aTextEdit_2 = QtWidgets.QLineEdit(parent=self.FitGroupConst_2)
        self.aTextEdit_2.setMinimumSize(QtCore.QSize(72, 23))
        self.aTextEdit_2.setMaximumSize(QtCore.QSize(72, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.aTextEdit_2.setFont(font)
        self.aTextEdit_2.setText("")
        self.aTextEdit_2.setObjectName("aTextEdit_2")
        self.horizontalLayout_45.addWidget(self.aTextEdit_2)
        self.gridLayout_6.addLayout(self.horizontalLayout_45, 0, 0, 1, 1)
        self.horizontalLayout_46 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_46.setObjectName("horizontalLayout_46")
        self.bLabel_6 = QtWidgets.QLabel(parent=self.FitGroupConst_2)
        self.bLabel_6.setMinimumSize(QtCore.QSize(50, 30))
        self.bLabel_6.setMaximumSize(QtCore.QSize(50, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.bLabel_6.setFont(font)
        self.bLabel_6.setObjectName("bLabel_6")
        self.horizontalLayout_46.addWidget(self.bLabel_6)
        self.bTextEdit_2 = QtWidgets.QLineEdit(parent=self.FitGroupConst_2)
        self.bTextEdit_2.setMinimumSize(QtCore.QSize(72, 23))
        self.bTextEdit_2.setMaximumSize(QtCore.QSize(72, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.bTextEdit_2.setFont(font)
        self.bTextEdit_2.setText("")
        self.bTextEdit_2.setObjectName("bTextEdit_2")
        self.horizontalLayout_46.addWidget(self.bTextEdit_2)
        self.gridLayout_6.addLayout(self.horizontalLayout_46, 1, 0, 1, 1)
        self.horizontalLayout_47 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_47.setObjectName("horizontalLayout_47")
        self.b1Label_6 = QtWidgets.QLabel(parent=self.FitGroupConst_2)
        self.b1Label_6.setMinimumSize(QtCore.QSize(50, 30))
        self.b1Label_6.setMaximumSize(QtCore.QSize(50, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.b1Label_6.setFont(font)
        self.b1Label_6.setObjectName("b1Label_6")
        self.horizontalLayout_47.addWidget(self.b1Label_6)
        self.b1TextEdit_2 = QtWidgets.QLineEdit(parent=self.FitGroupConst_2)
        self.b1TextEdit_2.setMinimumSize(QtCore.QSize(72, 23))
        self.b1TextEdit_2.setMaximumSize(QtCore.QSize(72, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.b1TextEdit_2.setFont(font)
        self.b1TextEdit_2.setText("")
        self.b1TextEdit_2.setObjectName("b1TextEdit_2")
        self.horizontalLayout_47.addWidget(self.b1TextEdit_2)
        self.gridLayout_6.addLayout(self.horizontalLayout_47, 2, 0, 1, 1)
        self.horizontalLayout_48 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_48.setObjectName("horizontalLayout_48")
        self.cLabel_6 = QtWidgets.QLabel(parent=self.FitGroupConst_2)
        self.cLabel_6.setMinimumSize(QtCore.QSize(50, 30))
        self.cLabel_6.setMaximumSize(QtCore.QSize(50, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.cLabel_6.setFont(font)
        self.cLabel_6.setObjectName("cLabel_6")
        self.horizontalLayout_48.addWidget(self.cLabel_6)
        self.cTextEdit_2 = QtWidgets.QLineEdit(parent=self.FitGroupConst_2)
        self.cTextEdit_2.setMinimumSize(QtCore.QSize(72, 23))
        self.cTextEdit_2.setMaximumSize(QtCore.QSize(72, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.cTextEdit_2.setFont(font)
        self.cTextEdit_2.setText("")
        self.cTextEdit_2.setObjectName("cTextEdit_2")
        self.horizontalLayout_48.addWidget(self.cTextEdit_2)
        self.gridLayout_6.addLayout(self.horizontalLayout_48, 3, 0, 1, 1)
        self.horizontalLayout_49 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_49.setObjectName("horizontalLayout_49")
        self.dLabel_6 = QtWidgets.QLabel(parent=self.FitGroupConst_2)
        self.dLabel_6.setMinimumSize(QtCore.QSize(50, 29))
        self.dLabel_6.setMaximumSize(QtCore.QSize(50, 29))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.dLabel_6.setFont(font)
        self.dLabel_6.setObjectName("dLabel_6")
        self.horizontalLayout_49.addWidget(self.dLabel_6)
        self.dTextEdit_2 = QtWidgets.QLineEdit(parent=self.FitGroupConst_2)
        self.dTextEdit_2.setMinimumSize(QtCore.QSize(72, 23))
        self.dTextEdit_2.setMaximumSize(QtCore.QSize(72, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.dTextEdit_2.setFont(font)
        self.dTextEdit_2.setText("")
        self.dTextEdit_2.setObjectName("dTextEdit_2")
        self.horizontalLayout_49.addWidget(self.dTextEdit_2)
        self.gridLayout_6.addLayout(self.horizontalLayout_49, 4, 0, 1, 1)
        self.horizontalLayout_50 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_50.setObjectName("horizontalLayout_50")
        self.eLabel_6 = QtWidgets.QLabel(parent=self.FitGroupConst_2)
        self.eLabel_6.setMinimumSize(QtCore.QSize(50, 30))
        self.eLabel_6.setMaximumSize(QtCore.QSize(50, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.eLabel_6.setFont(font)
        self.eLabel_6.setObjectName("eLabel_6")
        self.horizontalLayout_50.addWidget(self.eLabel_6)
        self.eTextEdit_2 = QtWidgets.QLineEdit(parent=self.FitGroupConst_2)
        self.eTextEdit_2.setMinimumSize(QtCore.QSize(72, 23))
        self.eTextEdit_2.setMaximumSize(QtCore.QSize(72, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.eTextEdit_2.setFont(font)
        self.eTextEdit_2.setText("")
        self.eTextEdit_2.setObjectName("eTextEdit_2")
        self.horizontalLayout_50.addWidget(self.eTextEdit_2)
        self.gridLayout_6.addLayout(self.horizontalLayout_50, 5, 0, 1, 1)
        self.horizontalLayout_51 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_51.setObjectName("horizontalLayout_51")
        self.fLabel_6 = QtWidgets.QLabel(parent=self.FitGroupConst_2)
        self.fLabel_6.setMinimumSize(QtCore.QSize(50, 30))
        self.fLabel_6.setMaximumSize(QtCore.QSize(50, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.fLabel_6.setFont(font)
        self.fLabel_6.setObjectName("fLabel_6")
        self.horizontalLayout_51.addWidget(self.fLabel_6)
        self.fTextEdit_2 = QtWidgets.QLineEdit(parent=self.FitGroupConst_2)
        self.fTextEdit_2.setMinimumSize(QtCore.QSize(72, 23))
        self.fTextEdit_2.setMaximumSize(QtCore.QSize(72, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.fTextEdit_2.setFont(font)
        self.fTextEdit_2.setText("")
        self.fTextEdit_2.setObjectName("fTextEdit_2")
        self.horizontalLayout_51.addWidget(self.fTextEdit_2)
        self.gridLayout_6.addLayout(self.horizontalLayout_51, 6, 0, 1, 1)
        self.horizontalLayout_52 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_52.setObjectName("horizontalLayout_52")
        self.fdLabel_6 = QtWidgets.QLabel(parent=self.FitGroupConst_2)
        self.fdLabel_6.setMinimumSize(QtCore.QSize(50, 30))
        self.fdLabel_6.setMaximumSize(QtCore.QSize(50, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.fdLabel_6.setFont(font)
        self.fdLabel_6.setObjectName("fdLabel_6")
        self.horizontalLayout_52.addWidget(self.fdLabel_6)
        self.fdTextEdit_2 = QtWidgets.QLineEdit(parent=self.FitGroupConst_2)
        self.fdTextEdit_2.setMinimumSize(QtCore.QSize(72, 23))
        self.fdTextEdit_2.setMaximumSize(QtCore.QSize(72, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.fdTextEdit_2.setFont(font)
        self.fdTextEdit_2.setText("")
        self.fdTextEdit_2.setObjectName("fdTextEdit_2")
        self.horizontalLayout_52.addWidget(self.fdTextEdit_2)
        self.gridLayout_6.addLayout(self.horizontalLayout_52, 7, 0, 1, 1)
        self.horizontalLayout_61.addWidget(self.FitGroupConst_2)
        self.FitGroupConst_4 = QtWidgets.QGroupBox(parent=self.FitTab)
        self.FitGroupConst_4.setMinimumSize(QtCore.QSize(150, 334))
        self.FitGroupConst_4.setMaximumSize(QtCore.QSize(150, 334))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.FitGroupConst_4.setFont(font)
        self.FitGroupConst_4.setObjectName("FitGroupConst_4")
        self.gridLayout_7 = QtWidgets.QGridLayout(self.FitGroupConst_4)
        self.gridLayout_7.setObjectName("gridLayout_7")
        self.horizontalLayout_53 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_53.setObjectName("horizontalLayout_53")
        self.aLabel_7 = QtWidgets.QLabel(parent=self.FitGroupConst_4)
        self.aLabel_7.setMinimumSize(QtCore.QSize(50, 30))
        self.aLabel_7.setMaximumSize(QtCore.QSize(50, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.aLabel_7.setFont(font)
        self.aLabel_7.setObjectName("aLabel_7")
        self.horizontalLayout_53.addWidget(self.aLabel_7)
        self.aTextEdit_4 = QtWidgets.QLineEdit(parent=self.FitGroupConst_4)
        self.aTextEdit_4.setMinimumSize(QtCore.QSize(72, 23))
        self.aTextEdit_4.setMaximumSize(QtCore.QSize(72, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.aTextEdit_4.setFont(font)
        self.aTextEdit_4.setText("")
        self.aTextEdit_4.setObjectName("aTextEdit_4")
        self.horizontalLayout_53.addWidget(self.aTextEdit_4)
        self.gridLayout_7.addLayout(self.horizontalLayout_53, 0, 0, 1, 1)
        self.horizontalLayout_54 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_54.setObjectName("horizontalLayout_54")
        self.bLabel_7 = QtWidgets.QLabel(parent=self.FitGroupConst_4)
        self.bLabel_7.setMinimumSize(QtCore.QSize(50, 30))
        self.bLabel_7.setMaximumSize(QtCore.QSize(50, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.bLabel_7.setFont(font)
        self.bLabel_7.setObjectName("bLabel_7")
        self.horizontalLayout_54.addWidget(self.bLabel_7)
        self.bTextEdit_4 = QtWidgets.QLineEdit(parent=self.FitGroupConst_4)
        self.bTextEdit_4.setMinimumSize(QtCore.QSize(72, 23))
        self.bTextEdit_4.setMaximumSize(QtCore.QSize(72, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.bTextEdit_4.setFont(font)
        self.bTextEdit_4.setText("")
        self.bTextEdit_4.setObjectName("bTextEdit_4")
        self.horizontalLayout_54.addWidget(self.bTextEdit_4)
        self.gridLayout_7.addLayout(self.horizontalLayout_54, 1, 0, 1, 1)
        self.horizontalLayout_55 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_55.setObjectName("horizontalLayout_55")
        self.b1Label_7 = QtWidgets.QLabel(parent=self.FitGroupConst_4)
        self.b1Label_7.setMinimumSize(QtCore.QSize(50, 30))
        self.b1Label_7.setMaximumSize(QtCore.QSize(50, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.b1Label_7.setFont(font)
        self.b1Label_7.setObjectName("b1Label_7")
        self.horizontalLayout_55.addWidget(self.b1Label_7)
        self.b1TextEdit_4 = QtWidgets.QLineEdit(parent=self.FitGroupConst_4)
        self.b1TextEdit_4.setMinimumSize(QtCore.QSize(72, 23))
        self.b1TextEdit_4.setMaximumSize(QtCore.QSize(72, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.b1TextEdit_4.setFont(font)
        self.b1TextEdit_4.setText("")
        self.b1TextEdit_4.setObjectName("b1TextEdit_4")
        self.horizontalLayout_55.addWidget(self.b1TextEdit_4)
        self.gridLayout_7.addLayout(self.horizontalLayout_55, 2, 0, 1, 1)
        self.horizontalLayout_56 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_56.setObjectName("horizontalLayout_56")
        self.cLabel_7 = QtWidgets.QLabel(parent=self.FitGroupConst_4)
        self.cLabel_7.setMinimumSize(QtCore.QSize(50, 30))
        self.cLabel_7.setMaximumSize(QtCore.QSize(50, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.cLabel_7.setFont(font)
        self.cLabel_7.setObjectName("cLabel_7")
        self.horizontalLayout_56.addWidget(self.cLabel_7)
        self.cTextEdit_4 = QtWidgets.QLineEdit(parent=self.FitGroupConst_4)
        self.cTextEdit_4.setMinimumSize(QtCore.QSize(72, 23))
        self.cTextEdit_4.setMaximumSize(QtCore.QSize(72, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.cTextEdit_4.setFont(font)
        self.cTextEdit_4.setText("")
        self.cTextEdit_4.setObjectName("cTextEdit_4")
        self.horizontalLayout_56.addWidget(self.cTextEdit_4)
        self.gridLayout_7.addLayout(self.horizontalLayout_56, 3, 0, 1, 1)
        self.horizontalLayout_57 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_57.setObjectName("horizontalLayout_57")
        self.dLabel_7 = QtWidgets.QLabel(parent=self.FitGroupConst_4)
        self.dLabel_7.setMinimumSize(QtCore.QSize(50, 29))
        self.dLabel_7.setMaximumSize(QtCore.QSize(50, 29))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.dLabel_7.setFont(font)
        self.dLabel_7.setObjectName("dLabel_7")
        self.horizontalLayout_57.addWidget(self.dLabel_7)
        self.dTextEdit_4 = QtWidgets.QLineEdit(parent=self.FitGroupConst_4)
        self.dTextEdit_4.setMinimumSize(QtCore.QSize(72, 23))
        self.dTextEdit_4.setMaximumSize(QtCore.QSize(72, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.dTextEdit_4.setFont(font)
        self.dTextEdit_4.setText("")
        self.dTextEdit_4.setObjectName("dTextEdit_4")
        self.horizontalLayout_57.addWidget(self.dTextEdit_4)
        self.gridLayout_7.addLayout(self.horizontalLayout_57, 4, 0, 1, 1)
        self.horizontalLayout_58 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_58.setObjectName("horizontalLayout_58")
        self.eLabel_7 = QtWidgets.QLabel(parent=self.FitGroupConst_4)
        self.eLabel_7.setMinimumSize(QtCore.QSize(50, 30))
        self.eLabel_7.setMaximumSize(QtCore.QSize(50, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.eLabel_7.setFont(font)
        self.eLabel_7.setObjectName("eLabel_7")
        self.horizontalLayout_58.addWidget(self.eLabel_7)
        self.eTextEdit_4 = QtWidgets.QLineEdit(parent=self.FitGroupConst_4)
        self.eTextEdit_4.setMinimumSize(QtCore.QSize(72, 23))
        self.eTextEdit_4.setMaximumSize(QtCore.QSize(72, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.eTextEdit_4.setFont(font)
        self.eTextEdit_4.setText("")
        self.eTextEdit_4.setObjectName("eTextEdit_4")
        self.horizontalLayout_58.addWidget(self.eTextEdit_4)
        self.gridLayout_7.addLayout(self.horizontalLayout_58, 5, 0, 1, 1)
        self.horizontalLayout_59 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_59.setObjectName("horizontalLayout_59")
        self.fLabel_7 = QtWidgets.QLabel(parent=self.FitGroupConst_4)
        self.fLabel_7.setMinimumSize(QtCore.QSize(50, 30))
        self.fLabel_7.setMaximumSize(QtCore.QSize(50, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.fLabel_7.setFont(font)
        self.fLabel_7.setObjectName("fLabel_7")
        self.horizontalLayout_59.addWidget(self.fLabel_7)
        self.fTextEdit_4 = QtWidgets.QLineEdit(parent=self.FitGroupConst_4)
        self.fTextEdit_4.setMinimumSize(QtCore.QSize(72, 23))
        self.fTextEdit_4.setMaximumSize(QtCore.QSize(72, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.fTextEdit_4.setFont(font)
        self.fTextEdit_4.setText("")
        self.fTextEdit_4.setObjectName("fTextEdit_4")
        self.horizontalLayout_59.addWidget(self.fTextEdit_4)
        self.gridLayout_7.addLayout(self.horizontalLayout_59, 6, 0, 1, 1)
        self.horizontalLayout_60 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_60.setObjectName("horizontalLayout_60")
        self.fdLabel_7 = QtWidgets.QLabel(parent=self.FitGroupConst_4)
        self.fdLabel_7.setMinimumSize(QtCore.QSize(50, 30))
        self.fdLabel_7.setMaximumSize(QtCore.QSize(50, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.fdLabel_7.setFont(font)
        self.fdLabel_7.setObjectName("fdLabel_7")
        self.horizontalLayout_60.addWidget(self.fdLabel_7)
        self.fdTextEdit_4 = QtWidgets.QLineEdit(parent=self.FitGroupConst_4)
        self.fdTextEdit_4.setMinimumSize(QtCore.QSize(72, 23))
        self.fdTextEdit_4.setMaximumSize(QtCore.QSize(72, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.fdTextEdit_4.setFont(font)
        self.fdTextEdit_4.setText("")
        self.fdTextEdit_4.setObjectName("fdTextEdit_4")
        self.horizontalLayout_60.addWidget(self.fdTextEdit_4)
        self.gridLayout_7.addLayout(self.horizontalLayout_60, 7, 0, 1, 1)
        self.horizontalLayout_61.addWidget(self.FitGroupConst_4)
        self.verticalLayout_37.addLayout(self.horizontalLayout_61)
        self.SimGroupConst = QtWidgets.QGroupBox(parent=self.FitTab)
        self.SimGroupConst.setMinimumSize(QtCore.QSize(644, 240))
        self.SimGroupConst.setMaximumSize(QtCore.QSize(620, 240))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.SimGroupConst.setFont(font)
        self.SimGroupConst.setObjectName("SimGroupConst")
        self.verticalLayout_11 = QtWidgets.QVBoxLayout(self.SimGroupConst)
        self.verticalLayout_11.setObjectName("verticalLayout_11")
        self.verticalLayout_10 = QtWidgets.QVBoxLayout()
        self.verticalLayout_10.setObjectName("verticalLayout_10")
        self.horizontalLayout_25 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_25.setObjectName("horizontalLayout_25")
        self.verticalLayout_24 = QtWidgets.QVBoxLayout()
        self.verticalLayout_24.setObjectName("verticalLayout_24")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.xaxixLabel = QtWidgets.QLabel(parent=self.SimGroupConst)
        self.xaxixLabel.setMinimumSize(QtCore.QSize(85, 23))
        self.xaxixLabel.setMaximumSize(QtCore.QSize(85, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.xaxixLabel.setFont(font)
        self.xaxixLabel.setObjectName("xaxixLabel")
        self.horizontalLayout.addWidget(self.xaxixLabel)
        self.xaxixTextEdit = QtWidgets.QLineEdit(parent=self.SimGroupConst)
        self.xaxixTextEdit.setMinimumSize(QtCore.QSize(50, 23))
        self.xaxixTextEdit.setMaximumSize(QtCore.QSize(50, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.xaxixTextEdit.setFont(font)
        self.xaxixTextEdit.setObjectName("xaxixTextEdit")
        self.horizontalLayout.addWidget(self.xaxixTextEdit)
        self.verticalLayout_24.addLayout(self.horizontalLayout)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.xaxixLabel_2 = QtWidgets.QLabel(parent=self.SimGroupConst)
        self.xaxixLabel_2.setMinimumSize(QtCore.QSize(85, 23))
        self.xaxixLabel_2.setMaximumSize(QtCore.QSize(85, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.xaxixLabel_2.setFont(font)
        self.xaxixLabel_2.setObjectName("xaxixLabel_2")
        self.horizontalLayout_2.addWidget(self.xaxixLabel_2)
        self.UsedCapTextEdit = QtWidgets.QLineEdit(parent=self.SimGroupConst)
        self.UsedCapTextEdit.setMinimumSize(QtCore.QSize(50, 23))
        self.UsedCapTextEdit.setMaximumSize(QtCore.QSize(50, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.UsedCapTextEdit.setFont(font)
        self.UsedCapTextEdit.setObjectName("UsedCapTextEdit")
        self.horizontalLayout_2.addWidget(self.UsedCapTextEdit)
        self.verticalLayout_24.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_20 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_20.setObjectName("horizontalLayout_20")
        self.DODLabel = QtWidgets.QLabel(parent=self.SimGroupConst)
        self.DODLabel.setMinimumSize(QtCore.QSize(85, 23))
        self.DODLabel.setMaximumSize(QtCore.QSize(85, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.DODLabel.setFont(font)
        self.DODLabel.setObjectName("DODLabel")
        self.horizontalLayout_20.addWidget(self.DODLabel)
        self.DODTextEdit = QtWidgets.QLineEdit(parent=self.SimGroupConst)
        self.DODTextEdit.setMinimumSize(QtCore.QSize(50, 23))
        self.DODTextEdit.setMaximumSize(QtCore.QSize(50, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.DODTextEdit.setFont(font)
        self.DODTextEdit.setObjectName("DODTextEdit")
        self.horizontalLayout_20.addWidget(self.DODTextEdit)
        self.verticalLayout_24.addLayout(self.horizontalLayout_20)
        spacerItem13 = QtWidgets.QSpacerItem(143, 44, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Expanding)
        self.verticalLayout_24.addItem(spacerItem13)
        self.horizontalLayout_25.addLayout(self.verticalLayout_24)
        self.verticalLayout_25 = QtWidgets.QVBoxLayout()
        self.verticalLayout_25.setObjectName("verticalLayout_25")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.CrateLabel = QtWidgets.QLabel(parent=self.SimGroupConst)
        self.CrateLabel.setMinimumSize(QtCore.QSize(84, 23))
        self.CrateLabel.setMaximumSize(QtCore.QSize(84, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.CrateLabel.setFont(font)
        self.CrateLabel.setObjectName("CrateLabel")
        self.horizontalLayout_3.addWidget(self.CrateLabel)
        self.CrateTextEdit = QtWidgets.QLineEdit(parent=self.SimGroupConst)
        self.CrateTextEdit.setMinimumSize(QtCore.QSize(50, 23))
        self.CrateTextEdit.setMaximumSize(QtCore.QSize(50, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.CrateTextEdit.setFont(font)
        self.CrateTextEdit.setObjectName("CrateTextEdit")
        self.horizontalLayout_3.addWidget(self.CrateTextEdit)
        self.verticalLayout_25.addLayout(self.horizontalLayout_3)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.SOCLabel = QtWidgets.QLabel(parent=self.SimGroupConst)
        self.SOCLabel.setMinimumSize(QtCore.QSize(84, 23))
        self.SOCLabel.setMaximumSize(QtCore.QSize(84, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.SOCLabel.setFont(font)
        self.SOCLabel.setObjectName("SOCLabel")
        self.horizontalLayout_4.addWidget(self.SOCLabel)
        self.SOCTextEdit = QtWidgets.QLineEdit(parent=self.SimGroupConst)
        self.SOCTextEdit.setMinimumSize(QtCore.QSize(50, 23))
        self.SOCTextEdit.setMaximumSize(QtCore.QSize(50, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.SOCTextEdit.setFont(font)
        self.SOCTextEdit.setObjectName("SOCTextEdit")
        self.horizontalLayout_4.addWidget(self.SOCTextEdit)
        self.verticalLayout_25.addLayout(self.horizontalLayout_4)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.CrateLabel_2 = QtWidgets.QLabel(parent=self.SimGroupConst)
        self.CrateLabel_2.setMinimumSize(QtCore.QSize(84, 23))
        self.CrateLabel_2.setMaximumSize(QtCore.QSize(84, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.CrateLabel_2.setFont(font)
        self.CrateLabel_2.setObjectName("CrateLabel_2")
        self.horizontalLayout_5.addWidget(self.CrateLabel_2)
        self.DcrateTextEdit = QtWidgets.QLineEdit(parent=self.SimGroupConst)
        self.DcrateTextEdit.setMinimumSize(QtCore.QSize(50, 23))
        self.DcrateTextEdit.setMaximumSize(QtCore.QSize(50, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.DcrateTextEdit.setFont(font)
        self.DcrateTextEdit.setObjectName("DcrateTextEdit")
        self.horizontalLayout_5.addWidget(self.DcrateTextEdit)
        self.verticalLayout_25.addLayout(self.horizontalLayout_5)
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.TempLabel = QtWidgets.QLabel(parent=self.SimGroupConst)
        self.TempLabel.setMinimumSize(QtCore.QSize(84, 23))
        self.TempLabel.setMaximumSize(QtCore.QSize(84, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.TempLabel.setFont(font)
        self.TempLabel.setObjectName("TempLabel")
        self.horizontalLayout_6.addWidget(self.TempLabel)
        self.TempTextEdit = QtWidgets.QLineEdit(parent=self.SimGroupConst)
        self.TempTextEdit.setMinimumSize(QtCore.QSize(50, 23))
        self.TempTextEdit.setMaximumSize(QtCore.QSize(50, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.TempTextEdit.setFont(font)
        self.TempTextEdit.setObjectName("TempTextEdit")
        self.horizontalLayout_6.addWidget(self.TempTextEdit)
        self.verticalLayout_25.addLayout(self.horizontalLayout_6)
        self.horizontalLayout_25.addLayout(self.verticalLayout_25)
        self.verticalLayout_26 = QtWidgets.QVBoxLayout()
        self.verticalLayout_26.setObjectName("verticalLayout_26")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.SOCLabel_3 = QtWidgets.QLabel(parent=self.SimGroupConst)
        self.SOCLabel_3.setMinimumSize(QtCore.QSize(85, 23))
        self.SOCLabel_3.setMaximumSize(QtCore.QSize(85, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.SOCLabel_3.setFont(font)
        self.SOCLabel_3.setObjectName("SOCLabel_3")
        self.horizontalLayout_7.addWidget(self.SOCLabel_3)
        self.SOCTextEdit_3 = QtWidgets.QLineEdit(parent=self.SimGroupConst)
        self.SOCTextEdit_3.setMinimumSize(QtCore.QSize(50, 23))
        self.SOCTextEdit_3.setMaximumSize(QtCore.QSize(50, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.SOCTextEdit_3.setFont(font)
        self.SOCTextEdit_3.setObjectName("SOCTextEdit_3")
        self.horizontalLayout_7.addWidget(self.SOCTextEdit_3)
        self.verticalLayout_26.addLayout(self.horizontalLayout_7)
        self.horizontalLayout_12 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_12.setObjectName("horizontalLayout_12")
        self.TempLabel_3 = QtWidgets.QLabel(parent=self.SimGroupConst)
        self.TempLabel_3.setMinimumSize(QtCore.QSize(85, 23))
        self.TempLabel_3.setMaximumSize(QtCore.QSize(85, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.TempLabel_3.setFont(font)
        self.TempLabel_3.setObjectName("TempLabel_3")
        self.horizontalLayout_12.addWidget(self.TempLabel_3)
        self.TempTextEdit_3 = QtWidgets.QLineEdit(parent=self.SimGroupConst)
        self.TempTextEdit_3.setMinimumSize(QtCore.QSize(50, 23))
        self.TempTextEdit_3.setMaximumSize(QtCore.QSize(50, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.TempTextEdit_3.setFont(font)
        self.TempTextEdit_3.setObjectName("TempTextEdit_3")
        self.horizontalLayout_12.addWidget(self.TempTextEdit_3)
        self.verticalLayout_26.addLayout(self.horizontalLayout_12)
        self.horizontalLayout_22 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_22.setObjectName("horizontalLayout_22")
        self.RestLabel_2 = QtWidgets.QLabel(parent=self.SimGroupConst)
        self.RestLabel_2.setMinimumSize(QtCore.QSize(85, 23))
        self.RestLabel_2.setMaximumSize(QtCore.QSize(85, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.RestLabel_2.setFont(font)
        self.RestLabel_2.setObjectName("RestLabel_2")
        self.horizontalLayout_22.addWidget(self.RestLabel_2)
        self.RestTextEdit_2 = QtWidgets.QLineEdit(parent=self.SimGroupConst)
        self.RestTextEdit_2.setMinimumSize(QtCore.QSize(50, 23))
        self.RestTextEdit_2.setMaximumSize(QtCore.QSize(50, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.RestTextEdit_2.setFont(font)
        self.RestTextEdit_2.setObjectName("RestTextEdit_2")
        self.horizontalLayout_22.addWidget(self.RestTextEdit_2)
        self.verticalLayout_26.addLayout(self.horizontalLayout_22)
        spacerItem14 = QtWidgets.QSpacerItem(143, 44, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Expanding)
        self.verticalLayout_26.addItem(spacerItem14)
        self.horizontalLayout_25.addLayout(self.verticalLayout_26)
        self.verticalLayout_28 = QtWidgets.QVBoxLayout()
        self.verticalLayout_28.setObjectName("verticalLayout_28")
        self.horizontalLayout_18 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_18.setObjectName("horizontalLayout_18")
        self.SOCLabel_2 = QtWidgets.QLabel(parent=self.SimGroupConst)
        self.SOCLabel_2.setMinimumSize(QtCore.QSize(84, 23))
        self.SOCLabel_2.setMaximumSize(QtCore.QSize(84, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.SOCLabel_2.setFont(font)
        self.SOCLabel_2.setObjectName("SOCLabel_2")
        self.horizontalLayout_18.addWidget(self.SOCLabel_2)
        self.SOCTextEdit_2 = QtWidgets.QLineEdit(parent=self.SimGroupConst)
        self.SOCTextEdit_2.setMinimumSize(QtCore.QSize(50, 23))
        self.SOCTextEdit_2.setMaximumSize(QtCore.QSize(50, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.SOCTextEdit_2.setFont(font)
        self.SOCTextEdit_2.setObjectName("SOCTextEdit_2")
        self.horizontalLayout_18.addWidget(self.SOCTextEdit_2)
        self.verticalLayout_28.addLayout(self.horizontalLayout_18)
        self.horizontalLayout_19 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_19.setObjectName("horizontalLayout_19")
        self.TempLabel_2 = QtWidgets.QLabel(parent=self.SimGroupConst)
        self.TempLabel_2.setMinimumSize(QtCore.QSize(84, 23))
        self.TempLabel_2.setMaximumSize(QtCore.QSize(84, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.TempLabel_2.setFont(font)
        self.TempLabel_2.setObjectName("TempLabel_2")
        self.horizontalLayout_19.addWidget(self.TempLabel_2)
        self.TempTextEdit_2 = QtWidgets.QLineEdit(parent=self.SimGroupConst)
        self.TempTextEdit_2.setMinimumSize(QtCore.QSize(50, 23))
        self.TempTextEdit_2.setMaximumSize(QtCore.QSize(50, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.TempTextEdit_2.setFont(font)
        self.TempTextEdit_2.setObjectName("TempTextEdit_2")
        self.horizontalLayout_19.addWidget(self.TempTextEdit_2)
        self.verticalLayout_28.addLayout(self.horizontalLayout_19)
        self.horizontalLayout_24 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_24.setObjectName("horizontalLayout_24")
        self.RestLabel = QtWidgets.QLabel(parent=self.SimGroupConst)
        self.RestLabel.setMinimumSize(QtCore.QSize(84, 23))
        self.RestLabel.setMaximumSize(QtCore.QSize(84, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.RestLabel.setFont(font)
        self.RestLabel.setObjectName("RestLabel")
        self.horizontalLayout_24.addWidget(self.RestLabel)
        self.RestTextEdit = QtWidgets.QLineEdit(parent=self.SimGroupConst)
        self.RestTextEdit.setMinimumSize(QtCore.QSize(50, 23))
        self.RestTextEdit.setMaximumSize(QtCore.QSize(50, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.RestTextEdit.setFont(font)
        self.RestTextEdit.setObjectName("RestTextEdit")
        self.horizontalLayout_24.addWidget(self.RestTextEdit)
        self.verticalLayout_28.addLayout(self.horizontalLayout_24)
        spacerItem15 = QtWidgets.QSpacerItem(142, 44, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Expanding)
        self.verticalLayout_28.addItem(spacerItem15)
        self.horizontalLayout_25.addLayout(self.verticalLayout_28)
        self.verticalLayout_10.addLayout(self.horizontalLayout_25)
        self.line_2 = QtWidgets.QFrame(parent=self.SimGroupConst)
        self.line_2.setMinimumSize(QtCore.QSize(620, 3))
        self.line_2.setMaximumSize(QtCore.QSize(598, 3))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.line_2.setFont(font)
        self.line_2.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.line_2.setObjectName("line_2")
        self.verticalLayout_10.addWidget(self.line_2)
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.FdLabel = QtWidgets.QLabel(parent=self.SimGroupConst)
        self.FdLabel.setMinimumSize(QtCore.QSize(80, 17))
        self.FdLabel.setMaximumSize(QtCore.QSize(80, 17))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.FdLabel.setFont(font)
        self.FdLabel.setObjectName("FdLabel")
        self.gridLayout_2.addWidget(self.FdLabel, 0, 0, 1, 1)
        self.FdLabel_2 = QtWidgets.QLabel(parent=self.SimGroupConst)
        self.FdLabel_2.setMinimumSize(QtCore.QSize(80, 17))
        self.FdLabel_2.setMaximumSize(QtCore.QSize(80, 17))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.FdLabel_2.setFont(font)
        self.FdLabel_2.setObjectName("FdLabel_2")
        self.gridLayout_2.addWidget(self.FdLabel_2, 0, 1, 1, 1)
        self.FdLabel_5 = QtWidgets.QLabel(parent=self.SimGroupConst)
        self.FdLabel_5.setMinimumSize(QtCore.QSize(80, 17))
        self.FdLabel_5.setMaximumSize(QtCore.QSize(80, 17))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.FdLabel_5.setFont(font)
        self.FdLabel_5.setObjectName("FdLabel_5")
        self.gridLayout_2.addWidget(self.FdLabel_5, 0, 2, 1, 1)
        self.FdLabel_6 = QtWidgets.QLabel(parent=self.SimGroupConst)
        self.FdLabel_6.setMinimumSize(QtCore.QSize(80, 17))
        self.FdLabel_6.setMaximumSize(QtCore.QSize(80, 17))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.FdLabel_6.setFont(font)
        self.FdLabel_6.setObjectName("FdLabel_6")
        self.gridLayout_2.addWidget(self.FdLabel_6, 0, 3, 1, 1)
        self.FdTextEdit_6 = QtWidgets.QLineEdit(parent=self.SimGroupConst)
        self.FdTextEdit_6.setMinimumSize(QtCore.QSize(80, 23))
        self.FdTextEdit_6.setMaximumSize(QtCore.QSize(80, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.FdTextEdit_6.setFont(font)
        self.FdTextEdit_6.setObjectName("FdTextEdit_6")
        self.gridLayout_2.addWidget(self.FdTextEdit_6, 1, 3, 1, 1)
        self.FdTextEdit_3 = QtWidgets.QLineEdit(parent=self.SimGroupConst)
        self.FdTextEdit_3.setMinimumSize(QtCore.QSize(80, 23))
        self.FdTextEdit_3.setMaximumSize(QtCore.QSize(80, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.FdTextEdit_3.setFont(font)
        self.FdTextEdit_3.setObjectName("FdTextEdit_3")
        self.gridLayout_2.addWidget(self.FdTextEdit_3, 1, 1, 1, 1)
        self.FdTextEdit = QtWidgets.QLineEdit(parent=self.SimGroupConst)
        self.FdTextEdit.setMinimumSize(QtCore.QSize(80, 23))
        self.FdTextEdit.setMaximumSize(QtCore.QSize(80, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.FdTextEdit.setFont(font)
        self.FdTextEdit.setObjectName("FdTextEdit")
        self.gridLayout_2.addWidget(self.FdTextEdit, 1, 0, 1, 1)
        self.FdLabel_3 = QtWidgets.QLabel(parent=self.SimGroupConst)
        self.FdLabel_3.setMinimumSize(QtCore.QSize(80, 17))
        self.FdLabel_3.setMaximumSize(QtCore.QSize(80, 17))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.FdLabel_3.setFont(font)
        self.FdLabel_3.setObjectName("FdLabel_3")
        self.gridLayout_2.addWidget(self.FdLabel_3, 0, 5, 1, 1)
        self.FdLabel_4 = QtWidgets.QLabel(parent=self.SimGroupConst)
        self.FdLabel_4.setMinimumSize(QtCore.QSize(80, 17))
        self.FdLabel_4.setMaximumSize(QtCore.QSize(80, 17))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.FdLabel_4.setFont(font)
        self.FdLabel_4.setObjectName("FdLabel_4")
        self.gridLayout_2.addWidget(self.FdLabel_4, 0, 4, 1, 1)
        self.FdTextEdit_5 = QtWidgets.QLineEdit(parent=self.SimGroupConst)
        self.FdTextEdit_5.setMinimumSize(QtCore.QSize(80, 23))
        self.FdTextEdit_5.setMaximumSize(QtCore.QSize(80, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.FdTextEdit_5.setFont(font)
        self.FdTextEdit_5.setObjectName("FdTextEdit_5")
        self.gridLayout_2.addWidget(self.FdTextEdit_5, 1, 2, 1, 1)
        self.FdTextEdit_2 = QtWidgets.QLineEdit(parent=self.SimGroupConst)
        self.FdTextEdit_2.setMinimumSize(QtCore.QSize(80, 23))
        self.FdTextEdit_2.setMaximumSize(QtCore.QSize(80, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.FdTextEdit_2.setFont(font)
        self.FdTextEdit_2.setObjectName("FdTextEdit_2")
        self.gridLayout_2.addWidget(self.FdTextEdit_2, 1, 4, 1, 1)
        self.FdTextEdit_4 = QtWidgets.QLineEdit(parent=self.SimGroupConst)
        self.FdTextEdit_4.setMinimumSize(QtCore.QSize(80, 23))
        self.FdTextEdit_4.setMaximumSize(QtCore.QSize(80, 23))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.FdTextEdit_4.setFont(font)
        self.FdTextEdit_4.setObjectName("FdTextEdit_4")
        self.gridLayout_2.addWidget(self.FdTextEdit_4, 1, 5, 1, 1)
        self.verticalLayout_10.addLayout(self.gridLayout_2)
        self.verticalLayout_11.addLayout(self.verticalLayout_10)
        self.verticalLayout_37.addWidget(self.SimGroupConst)
        self.horizontalLayout_170.addLayout(self.verticalLayout_37)
        self.real_cycle_simul_tab = QtWidgets.QTabWidget(parent=self.FitTab)
        self.real_cycle_simul_tab.setMinimumSize(QtCore.QSize(1200, 830))
        self.real_cycle_simul_tab.setMaximumSize(QtCore.QSize(1200, 830))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.real_cycle_simul_tab.setFont(font)
        self.real_cycle_simul_tab.setObjectName("real_cycle_simul_tab")
        self.horizontalLayout_170.addWidget(self.real_cycle_simul_tab)
        self.horizontalLayout_171.addLayout(self.horizontalLayout_170)
        self.line_6 = QtWidgets.QFrame(parent=self.FitTab)
        self.line_6.setMinimumSize(QtCore.QSize(0, 3))
        self.line_6.setMaximumSize(QtCore.QSize(0, 3))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.line_6.setFont(font)
        self.line_6.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        self.line_6.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.line_6.setObjectName("line_6")
        self.horizontalLayout_171.addWidget(self.line_6)
        self.tabWidget.addTab(self.FitTab, "")
        self.verticalLayout_38.addWidget(self.tabWidget)
        self.horizontalLayout_23 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_23.setObjectName("horizontalLayout_23")
        self.mount_toyo = QtWidgets.QPushButton(parent=self.layoutWidget)
        self.mount_toyo.setMinimumSize(QtCore.QSize(200, 40))
        self.mount_toyo.setMaximumSize(QtCore.QSize(200, 40))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.mount_toyo.setFont(font)
        self.mount_toyo.setObjectName("mount_toyo")
        self.horizontalLayout_23.addWidget(self.mount_toyo)
        self.mount_pne_1 = QtWidgets.QPushButton(parent=self.layoutWidget)
        self.mount_pne_1.setMinimumSize(QtCore.QSize(200, 40))
        self.mount_pne_1.setMaximumSize(QtCore.QSize(200, 40))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.mount_pne_1.setFont(font)
        self.mount_pne_1.setObjectName("mount_pne_1")
        self.horizontalLayout_23.addWidget(self.mount_pne_1)
        self.line = QtWidgets.QFrame(parent=self.layoutWidget)
        self.line.setMinimumSize(QtCore.QSize(3, 35))
        self.line.setMaximumSize(QtCore.QSize(3, 35))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.line.setFont(font)
        self.line.setFrameShape(QtWidgets.QFrame.Shape.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.line.setObjectName("line")
        self.horizontalLayout_23.addWidget(self.line)
        self.mount_pne_2 = QtWidgets.QPushButton(parent=self.layoutWidget)
        self.mount_pne_2.setMinimumSize(QtCore.QSize(200, 40))
        self.mount_pne_2.setMaximumSize(QtCore.QSize(200, 40))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.mount_pne_2.setFont(font)
        self.mount_pne_2.setObjectName("mount_pne_2")
        self.horizontalLayout_23.addWidget(self.mount_pne_2)
        self.mount_pne_3 = QtWidgets.QPushButton(parent=self.layoutWidget)
        self.mount_pne_3.setMinimumSize(QtCore.QSize(200, 40))
        self.mount_pne_3.setMaximumSize(QtCore.QSize(200, 40))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.mount_pne_3.setFont(font)
        self.mount_pne_3.setObjectName("mount_pne_3")
        self.horizontalLayout_23.addWidget(self.mount_pne_3)
        self.mount_pne_4 = QtWidgets.QPushButton(parent=self.layoutWidget)
        self.mount_pne_4.setMinimumSize(QtCore.QSize(200, 40))
        self.mount_pne_4.setMaximumSize(QtCore.QSize(200, 40))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.mount_pne_4.setFont(font)
        self.mount_pne_4.setObjectName("mount_pne_4")
        self.horizontalLayout_23.addWidget(self.mount_pne_4)
        self.line_12 = QtWidgets.QFrame(parent=self.layoutWidget)
        self.line_12.setMinimumSize(QtCore.QSize(3, 35))
        self.line_12.setMaximumSize(QtCore.QSize(3, 35))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.line_12.setFont(font)
        self.line_12.setFrameShape(QtWidgets.QFrame.Shape.VLine)
        self.line_12.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.line_12.setObjectName("line_12")
        self.horizontalLayout_23.addWidget(self.line_12)
        self.mount_pne_5 = QtWidgets.QPushButton(parent=self.layoutWidget)
        self.mount_pne_5.setMinimumSize(QtCore.QSize(200, 40))
        self.mount_pne_5.setMaximumSize(QtCore.QSize(200, 40))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.mount_pne_5.setFont(font)
        self.mount_pne_5.setObjectName("mount_pne_5")
        self.horizontalLayout_23.addWidget(self.mount_pne_5)
        self.line_9 = QtWidgets.QFrame(parent=self.layoutWidget)
        self.line_9.setMinimumSize(QtCore.QSize(3, 35))
        self.line_9.setMaximumSize(QtCore.QSize(3, 35))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.line_9.setFont(font)
        self.line_9.setFrameShape(QtWidgets.QFrame.Shape.VLine)
        self.line_9.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.line_9.setObjectName("line_9")
        self.horizontalLayout_23.addWidget(self.line_9)
        self.mount_all = QtWidgets.QPushButton(parent=self.layoutWidget)
        self.mount_all.setMinimumSize(QtCore.QSize(200, 40))
        self.mount_all.setMaximumSize(QtCore.QSize(200, 40))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.mount_all.setFont(font)
        self.mount_all.setObjectName("mount_all")
        self.horizontalLayout_23.addWidget(self.mount_all)
        self.line_5 = QtWidgets.QFrame(parent=self.layoutWidget)
        self.line_5.setMinimumSize(QtCore.QSize(3, 35))
        self.line_5.setMaximumSize(QtCore.QSize(3, 35))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.line_5.setFont(font)
        self.line_5.setFrameShape(QtWidgets.QFrame.Shape.VLine)
        self.line_5.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.line_5.setObjectName("line_5")
        self.horizontalLayout_23.addWidget(self.line_5)
        self.unmount_all = QtWidgets.QPushButton(parent=self.layoutWidget)
        self.unmount_all.setMinimumSize(QtCore.QSize(200, 40))
        self.unmount_all.setMaximumSize(QtCore.QSize(200, 40))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.unmount_all.setFont(font)
        self.unmount_all.setObjectName("unmount_all")
        self.horizontalLayout_23.addWidget(self.unmount_all)
        self.verticalLayout_38.addLayout(self.horizontalLayout_23)
        self.verticalLayout_39.addLayout(self.verticalLayout_38)
        self.line_7 = QtWidgets.QFrame(parent=self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.line_7.setFont(font)
        self.line_7.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        self.line_7.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.line_7.setObjectName("line_7")
        self.verticalLayout_39.addWidget(self.line_7)
        self.horizontalLayout_13 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_13.setObjectName("horizontalLayout_13")
        self.saveok = QtWidgets.QCheckBox(parent=self.layoutWidget)
        self.saveok.setMinimumSize(QtCore.QSize(100, 30))
        self.saveok.setMaximumSize(QtCore.QSize(100, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.saveok.setFont(font)
        self.saveok.setObjectName("saveok")
        self.horizontalLayout_13.addWidget(self.saveok)
        self.ect_saveok = QtWidgets.QCheckBox(parent=self.layoutWidget)
        self.ect_saveok.setMinimumSize(QtCore.QSize(150, 30))
        self.ect_saveok.setMaximumSize(QtCore.QSize(150, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.ect_saveok.setFont(font)
        self.ect_saveok.setObjectName("ect_saveok")
        self.horizontalLayout_13.addWidget(self.ect_saveok)
        self.figsaveok = QtWidgets.QCheckBox(parent=self.layoutWidget)
        self.figsaveok.setMinimumSize(QtCore.QSize(100, 30))
        self.figsaveok.setMaximumSize(QtCore.QSize(100, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.figsaveok.setFont(font)
        self.figsaveok.setObjectName("figsaveok")
        self.horizontalLayout_13.addWidget(self.figsaveok)
        self.progressBar = QtWidgets.QProgressBar(parent=self.layoutWidget)
        self.progressBar.setMinimumSize(QtCore.QSize(1400, 30))
        self.progressBar.setMaximumSize(QtCore.QSize(1400, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(9)
        self.progressBar.setFont(font)
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.horizontalLayout_13.addWidget(self.progressBar)
        self.verticalLayout_39.addLayout(self.horizontalLayout_13)

        self.retranslateUi(sitool)
        self.tabWidget.setCurrentIndex(1)
        self.tabWidget_2.setCurrentIndex(0)
        self.cycle_tab.setCurrentIndex(-1)
        self.set_tab.setCurrentIndex(-1)
        self.dvdq_simul_tab.setCurrentIndex(-1)
        self.cycle_simul_tab_eu.setCurrentIndex(-1)
        self.cycle_simul_tab.setCurrentIndex(-1)
        self.real_cycle_simul_tab.setCurrentIndex(-1)
        QtCore.QMetaObject.connectSlotsByName(sitool)

    def retranslateUi(self, sitool):
        _translate = QtCore.QCoreApplication.translate
        sitool.setWindowTitle(_translate("sitool", "BatteryDataTool v251103"))
        self.tb_room.setItemText(0, _translate("sitool", "R5 15F"))
        self.tb_room.setItemText(1, _translate("sitool", "R5 3F B-1"))
        self.tb_room.setItemText(2, _translate("sitool", "R5 3F B-2"))
        self.tb_room.setItemText(3, _translate("sitool", "R5 3F A"))
        self.tb_room.setItemText(4, _translate("sitool", "전체"))
        self.tb_cycler.setItemText(0, _translate("sitool", "Toyo1"))
        self.tb_cycler.setItemText(1, _translate("sitool", "Toyo2"))
        self.tb_cycler.setItemText(2, _translate("sitool", "Toyo3"))
        self.tb_cycler.setItemText(3, _translate("sitool", "Toyo4"))
        self.tb_cycler.setItemText(4, _translate("sitool", "Toyo5"))
        self.tb_cycler.setItemText(5, _translate("sitool", "PNE1"))
        self.tb_cycler.setItemText(6, _translate("sitool", "PNE2"))
        self.tb_cycler.setItemText(7, _translate("sitool", "PNE3"))
        self.tb_cycler.setItemText(8, _translate("sitool", "PNE4"))
        self.tb_cycler.setItemText(9, _translate("sitool", "PNE5"))
        self.tb_info.setItemText(0, _translate("sitool", "채널"))
        self.tb_info.setItemText(1, _translate("sitool", "파일명"))
        self.tb_info.setItemText(2, _translate("sitool", "스케쥴명"))
        self.tb_info.setItemText(3, _translate("sitool", "날짜"))
        self.tb_info.setItemText(4, _translate("sitool", "파트"))
        self.tb_info.setItemText(5, _translate("sitool", "이름"))
        self.tb_info.setItemText(6, _translate("sitool", "온도"))
        self.tb_info.setItemText(7, _translate("sitool", "현재상태 (현재 Step / 현재 Loop Cycle / 총 Cycle)"))
        self.tb_info.setItemText(8, _translate("sitool", "현재 전압"))
        self.tb_info.setItemText(9, _translate("sitool", "셀 경로"))
        self.label_9.setText(_translate("sitool", "강조할 문자"))
        item = self.tb_summary.verticalHeaderItem(0)
        item.setText(_translate("sitool", "사용 가능"))
        item = self.tb_summary.verticalHeaderItem(1)
        item.setText(_translate("sitool", "사용 중"))
        __sortingEnabled = self.tb_summary.isSortingEnabled()
        self.tb_summary.setSortingEnabled(False)
        self.tb_summary.setSortingEnabled(__sortingEnabled)
        __sortingEnabled = self.tableWidget.isSortingEnabled()
        self.tableWidget.setSortingEnabled(False)
        item = self.tableWidget.item(0, 0)
        item.setText(_translate("sitool", "온도별"))
        item = self.tableWidget.item(0, 1)
        item.setText(_translate("sitool", "코인셀"))
        item = self.tableWidget.item(0, 2)
        item.setText(_translate("sitool", "15도 채널"))
        item = self.tableWidget.item(0, 3)
        item.setText(_translate("sitool", "23도 채널"))
        item = self.tableWidget.item(0, 4)
        item.setText(_translate("sitool", "35도 채널"))
        item = self.tableWidget.item(0, 5)
        item.setText(_translate("sitool", "45도 채널"))
        item = self.tableWidget.item(1, 0)
        item.setText(_translate("sitool", "TOYO"))
        item = self.tableWidget.item(1, 1)
        item.setText(_translate("sitool", "완료, 작업정지"))
        item = self.tableWidget.item(1, 2)
        item.setText(_translate("sitool", "완료, 작업정지 (셀 있음)"))
        item = self.tableWidget.item(2, 0)
        item.setText(_translate("sitool", "PNE"))
        item = self.tableWidget.item(2, 1)
        item.setText(_translate("sitool", "대기 (셀없음)"))
        item = self.tableWidget.item(2, 2)
        item.setText(_translate("sitool", "완료 (셀 있음)"))
        item = self.tableWidget.item(2, 3)
        item.setText(_translate("sitool", "작업멈춤"))
        item = self.tableWidget.item(2, 5)
        item.setText(_translate("sitool", "강조할 문자 필터"))
        self.tableWidget.setSortingEnabled(__sortingEnabled)
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("sitool", "현황"))
        self.chk_cyclepath.setText(_translate("sitool", "지정Path사용"))
        self.chk_ectpath.setText(_translate("sitool", "ECT path 사용"))
        self.cycle_tab_reset.setText(_translate("sitool", "Tab Reset"))
        self.capacitygroup.setTitle(_translate("sitool", "용량 선택"))
        self.inicaprate.setText(_translate("sitool", "1) Cyclepath 이름 용량 기준\n"
"2) 테스트 이름 용량 기준\n"
"3) 첫 Cycle의 Crate 기준"))
        self.ratetext.setText(_translate("sitool", "0.2"))
        self.inicaptype.setText(_translate("sitool", "용량값 직접 입력"))
        self.capacitytext.setText(_translate("sitool", "58"))
        self.dcirchk.setText(_translate("sitool", "PNE 설비 DCIR (SOC100 10s 방전 Pulse)"))
        self.pulsedcir.setText(_translate("sitool", "PNE 10s DCIR (SOC5, 50 10s 방전 Pulse)"))
        self.mkdcir.setText(_translate("sitool", "PNE DCIR (SOC 30/50/70 충전, SOC 70/50/30 방전, 1s Pulse/RSS)\n"
"PNE DCIR2 (SOC 70/50/30/15 방전, 1s Pulse/RSS) - 그래프 방전 70%"))
        self.dcirchk_2.setText(_translate("sitool", "DCIR 고정 해제"))
        self.cycxlabel_2.setText(_translate("sitool", "  Y축 최대"))
        self.tcyclerngyhl.setText(_translate("sitool", "1.10"))
        self.cycxlabel_3.setText(_translate("sitool", "  Y축 최소"))
        self.tcyclerngyll.setText(_translate("sitool", "0.65"))
        self.cycxlabel.setText(_translate("sitool", "  X축 최대"))
        self.tcyclerng.setText(_translate("sitool", "0"))
        self.dcirscalelb.setText(_translate("sitool", "  DCIR scale 늘리기 (x ?배)"))
        self.dcirscale.setText(_translate("sitool", "0"))
        self.indiv_cycle.setText(_translate("sitool", "개별 Cycle"))
        self.overall_cycle.setText(_translate("sitool", "통합 Cycle"))
        self.link_cycle.setText(_translate("sitool", "연결 Cycle"))
        self.AppCycConfirm.setText(_translate("sitool", "신뢰성 Cycle"))
        self.link_cycle_indiv.setText(_translate("sitool", "연결 Cycle  \n"
" 여러개 개별"))
        self.link_cycle_overall.setText(_translate("sitool", "연결 Cycle \n"
" 여러개 통합"))
        self.tabWidget_2.setTabText(self.tabWidget_2.indexOf(self.tab_5), _translate("sitool", "Cycle"))
        self.CycProfile.setText(_translate("sitool", "사이클 통합"))
        self.CellProfile.setText(_translate("sitool", "셀별 통합"))
        self.chk_dqdv.setText(_translate("sitool", "dQdV X/Y축 변환"))
        self.stepnumlb.setText(_translate("sitool", "Cycle\n"
"(원하는 스텝들을 띄어쓰기나 -로 표기)\n"
"예) 3 4 5 8-9"))
        self.stepnum.setPlainText(_translate("sitool", "2"))
        self.smoothlb_3.setText(_translate("sitool", "전압 Y축 하한"))
        self.volrngyhl.setText(_translate("sitool", "2.5"))
        self.smoothlb_2.setText(_translate("sitool", "전압 Y축 상한"))
        self.volrngyll.setText(_translate("sitool", "4.7"))
        self.smoothlb_4.setText(_translate("sitool", "전압 Y축 간격"))
        self.volrnggap.setText(_translate("sitool", "0.1"))
        self.smoothlb.setText(_translate("sitool", "Smooth (0 이면 자동)"))
        self.smooth.setText(_translate("sitool", "0"))
        self.cutofflb.setText(_translate("sitool", "컷오프 (전류,전압)"))
        self.cutoff.setText(_translate("sitool", "0"))
        self.dqdvscalelb.setText(_translate("sitool", "dQdV 축늘리기"))
        self.dqdvscale.setText(_translate("sitool", "1"))
        self.StepConfirm.setText(_translate("sitool", "충전 Step 확인"))
        self.ChgConfirm.setText(_translate("sitool", "충전 분석"))
        self.RateConfirm.setText(_translate("sitool", "율별 충전 확인"))
        self.DchgConfirm.setText(_translate("sitool", "방전 분석"))
        self.ContinueConfirm.setText(_translate("sitool", "HPPC/ GITT/ ECT"))
        self.DCIRConfirm.setText(_translate("sitool", "DCIR"))
        self.tabWidget_2.setTabText(self.tabWidget_2.indexOf(self.tab_6), _translate("sitool", "Profile"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.CycTab), _translate("sitool", "사이클데이터"))
        self.label_10.setText(_translate("sitool", "패턴 리스트 Load를 누르고 패턴을 선택한 후에 오른쪽 패널에서 원하는 부분 수정"))
        self.chk_coincell.setText(_translate("sitool", "Coin cell"))
        self.cycxlabel_7.setText(_translate("sitool", "패턴 경로"))
        self.ptn_load.setText(_translate("sitool", "패턴 리스트 Load"))
        self.ptn_ori_path.setText(_translate("sitool", "C:\\Program Files\\PNE CTSPro\\DataBase\\Cycler_Schedule_2000.mdb"))
        self.groupBox_8.setTitle(_translate("sitool", "전류 일괄 변경"))
        self.cycxlabel_5.setText(_translate("sitool", "방전 최대 C-rate"))
        self.ptn_crate.setText(_translate("sitool", "1.0"))
        self.cycxlabel_6.setText(_translate("sitool", "변경할 용량 기준(mAh)"))
        self.ptn_capacity.setText(_translate("sitool", "4000"))
        self.chg_ptn.setText(_translate("sitool", "전류 일괄 변경"))
        self.groupBox_5.setTitle(_translate("sitool", "충전 CV 전압 변경 (mV)"))
        self.cycxlabel_19.setText(_translate("sitool", "기존 전압 (mV)"))
        self.ptn_chgv_pre.setText(_translate("sitool", "4470"))
        self.cycxlabel_20.setText(_translate("sitool", "변경할 전압(mV)"))
        self.ptn_chgv_after.setText(_translate("sitool", "4500"))
        self.chg_ptn_chgv.setText(_translate("sitool", "충전 전압 바꾸기"))
        self.groupBox_2.setTitle(_translate("sitool", "충방전 전류 변경 (mA)"))
        self.cycxlabel_9.setText(_translate("sitool", "기존 전류 (mA)"))
        self.ptn_refi_pre.setText(_translate("sitool", "2000"))
        self.cycxlabel_10.setText(_translate("sitool", "변경할 전류(mA)"))
        self.ptn_refi_after.setText(_translate("sitool", "1000"))
        self.chg_ptn_refi.setText(_translate("sitool", "충방전 전류 바꾸기"))
        self.groupBox_7.setTitle(_translate("sitool", "방전 CV 전압 변경 (mV)"))
        self.cycxlabel_27.setText(_translate("sitool", "기존 전압 (mV)"))
        self.ptn_dchgv_pre.setText(_translate("sitool", "3000"))
        self.cycxlabel_28.setText(_translate("sitool", "변경할 전압(mV)"))
        self.ptn_dchgv_after.setText(_translate("sitool", "2750"))
        self.chg_ptn_dchgv.setText(_translate("sitool", "방전 전압 바꾸기"))
        self.groupBox_3.setTitle(_translate("sitool", "Cut-off 전류 변경 (mA)"))
        self.cycxlabel_11.setText(_translate("sitool", "기존 전류 (mA)"))
        self.ptn_endi_pre.setText(_translate("sitool", "100"))
        self.cycxlabel_13.setText(_translate("sitool", "변경할 전류(mA)"))
        self.ptn_endi_after.setText(_translate("sitool", "50"))
        self.chg_ptn_endi.setText(_translate("sitool", "종지 전류 바꾸기"))
        self.groupBox_6.setTitle(_translate("sitool", "Cut-off 전압 변경 (mV)"))
        self.cycxlabel_21.setText(_translate("sitool", "기존 전압 (mV)"))
        self.ptn_endv_pre.setText(_translate("sitool", "3000"))
        self.cycxlabel_22.setText(_translate("sitool", "변경할 전압(mV)"))
        self.ptn_endv_after.setText(_translate("sitool", "3300"))
        self.chg_ptn_endv.setText(_translate("sitool", "종지 전압 바꾸기"))
        self.groupBox_4.setTitle(_translate("sitool", "Step 변경 (전체 밀기)"))
        self.cycxlabel_16.setText(_translate("sitool", "변경할 스텝"))
        self.ptn_step_pre.setText(_translate("sitool", "10"))
        self.cycxlabel_14.setText(_translate("sitool", "추가로 더할 스텝"))
        self.ptn_step_after.setText(_translate("sitool", "5"))
        self.chg_ptn_step.setText(_translate("sitool", "Step 바꾸기"))
        self.ptn_list.setSortingEnabled(False)
        item = self.ptn_list.horizontalHeaderItem(0)
        item.setText(_translate("sitool", "패턴폴더"))
        item = self.ptn_list.horizontalHeaderItem(1)
        item.setText(_translate("sitool", "패턴이름"))
        item = self.ptn_list.horizontalHeaderItem(2)
        item.setText(_translate("sitool", "비고"))
        item = self.ptn_list.horizontalHeaderItem(3)
        item.setText(_translate("sitool", "TestID"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("sitool", "패턴수정"))
        self.Capacitynum.setTitle(_translate("sitool", "용량기준 (mAh)"))
        self.SetMincapacity.setText(_translate("sitool", "4565"))
        self.Capacitynum_2.setTitle(_translate("sitool", "Max Cycle"))
        self.SetMaxCycle.setText(_translate("sitool", "0"))
        self.gCyclesetting.setTitle(_translate("sitool", "Cycle 세팅 관련"))
        self.realcyc.setText(_translate("sitool", "실제 사이클"))
        self.resetcycle.setText(_translate("sitool", "보정사이클"))
        self.cycxlabel_4.setText(_translate("sitool", "X축 최대"))
        self.setcyclexscale.setText(_translate("sitool", "0"))
        self.allcycle.setText(_translate("sitool", "전체 사이클"))
        self.recentcycle.setText(_translate("sitool", "최근 사이클"))
        self.recentcycleno.setText(_translate("sitool", "20"))
        self.manualcycle.setText(_translate("sitool", "지정 사이클 \n"
" (0, 0이면 최소, 최대 기준)"))
        self.manualcycleno.setText(_translate("sitool", "0 0"))
        self.groupBox.setTitle(_translate("sitool", "ECT 세팅 관련"))
        self.label_3.setText(_translate("sitool", "Max_capacity (SOC를 용량으로 환산 결과)"))
        self.label_11.setText(_translate("sitool", "SET off 전압 기준 (45s Avg, V)"))
        self.label_12.setText(_translate("sitool", "SOC 오차 Max"))
        self.label_14.setText(_translate("sitool", "SOC 오차 Avg"))
        self.label_13.setText(_translate("sitool", "ECT SOC 오차 Max"))
        self.label_15.setText(_translate("sitool", "ECT SOC 오차 Avg"))
        self.label_4.setText(_translate("sitool", "Battery Log 프로그램 결과 확인"))
        self.label_7.setText(_translate("sitool", "ECT 프로그램(ChemBatt) 결과 확인"))
        self.SETTabReset.setText(_translate("sitool", "Tab Reset"))
        self.ECTSOC.setText(_translate("sitool", "ECT SOC"))
        self.SetlogConfirm.setText(_translate("sitool", "Battery Dump Profile"))
        self.ECTShort.setText(_translate("sitool", "ECT Short"))
        self.SetConfirm.setText(_translate("sitool", "Battery Status Profile"))
        self.ECTSetProfile.setText(_translate("sitool", "ECT profile"))
        self.SetCycle.setText(_translate("sitool", "Battery Status Cycle"))
        self.ECTSetCycle.setText(_translate("sitool", "ECT cycle"))
        self.ECTSetlog2.setText(_translate("sitool", "ECT log"))
        self.ECTSetlog.setText(_translate("sitool", "ECT log vs App"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.SetTab), _translate("sitool", "세트 결과"))
        self.cycxlabel_29.setText(_translate("sitool", "양극 profile 경로"))
        self.cycxlabel_8.setText(_translate("sitool", "음극 profile 경로"))
        self.cycxlabel_12.setText(_translate("sitool", "실측 profile 경로"))
        self.cycxlabel_62.setText(_translate("sitool", "시작 기준"))
        self.dvdq_start_soc.setText(_translate("sitool", "7"))
        self.cycxlabel_61.setText(_translate("sitool", "끝 기준"))
        self.dvdq_end_soc.setText(_translate("sitool", "100"))
        self.cycxlabel_60.setText(_translate("sitool", "Smoothing 기준"))
        self.dvdq_full_smoothing_no.setText(_translate("sitool", "500"))
        self.cycxlabel_32.setText(_translate("sitool", "셀 총용량"))
        self.cycxlabel_30.setText(_translate("sitool", "양극 총용량"))
        self.cycxlabel_31.setText(_translate("sitool", "음극 총용량"))
        self.cycxlabel_18.setText(_translate("sitool", "양극 mass"))
        self.cycxlabel_23.setText(_translate("sitool", "양극 Slip"))
        self.cycxlabel_24.setText(_translate("sitool", "음극 mass"))
        self.cycxlabel_25.setText(_translate("sitool", "음극 Slip"))
        self.cycxlabel_15.setText(_translate("sitool", "실행 횟수"))
        self.dvdq_test_no.setText(_translate("sitool", "100"))
        self.cycxlabel_26.setText(_translate("sitool", "RMS (%)"))
        self.dvdq_ini_reset.setText(_translate("sitool", "초기치 Reset"))
        self.mat_dvdq_btn.setText(_translate("sitool", "1) 소재 결과 Load"))
        self.pro_dvdq_btn.setText(_translate("sitool", "2) 실험 결과 Load"))
        self.dvdq_tab_reset.setText(_translate("sitool", "Tab Reset"))
        self.dvdq_fitting.setText(_translate("sitool", "dVdQ 자동 Fitting 실행"))
        self.dvdq_fitting_2.setText(_translate("sitool", "dVdQ 수동 조절"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.dvdq), _translate("sitool", "dVdQ 분석"))
        self.cycxlabel_54.setText(_translate("sitool", "Load Cycle parameter 경로"))
        self.TabReset_eu.setText(_translate("sitool", "Tab Reset"))
        self.aLabel_4.setText(_translate("sitool", "  Parameter Setting"))
        self.fix_swelling_eu.setText(_translate("sitool", "Swelling 예측"))
        self.cycxlabel_38.setText(_translate("sitool", "   수식:     1 - exp(a * T + b) *(N * f_d) ^ b1 - exp(c * T + d) * (N * f_d) ^ (e * T + f)"))
        self.aLabel_3.setText(_translate("sitool", "   a"))
        self.bLabel_3.setText(_translate("sitool", "   b"))
        self.b1Label_3.setText(_translate("sitool", "   b1"))
        self.cLabel_3.setText(_translate("sitool", "   c"))
        self.dLabel_3.setText(_translate("sitool", "   d"))
        self.eLabel_3.setText(_translate("sitool", "   e"))
        self.fLabel_3.setText(_translate("sitool", "   f"))
        self.fdLabel_3.setText(_translate("sitool", "   f_d"))
        self.fdLabel_4.setText(_translate("sitool", "   const_f_d"))
        self.cycxlabel_36.setText(_translate("sitool", "  Y축 최대"))
        self.simul_y_max_eu.setText(_translate("sitool", "1.0"))
        self.cycxlabel_35.setText(_translate("sitool", "  Y축 최소"))
        self.simul_y_min_eu.setText(_translate("sitool", "0.8"))
        self.cycxlabel_37.setText(_translate("sitool", "  X축 최대"))
        self.simul_x_max_eu.setText(_translate("sitool", "5000"))
        self.ParameterReset_eu.setText(_translate("sitool", "Parameter Reset"))
        self.load_cycparameter_eu.setText(_translate("sitool", "Parameter Load"))
        self.save_cycparameter_eu.setText(_translate("sitool", "Parameter Save"))
        self.FitConfirm_eu.setText(_translate("sitool", "변수 계산"))
        self.ConstFitConfirm_eu.setText(_translate("sitool", "변수 고정"))
        self.indivConstFitConfirm_eu.setText(_translate("sitool", "개별 결과 변수 고정"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_4), _translate("sitool", "Eu 수명 예측"))
        self.cycxlabel_52.setText(_translate("sitool", "02C parameter 경로"))
        self.cycxlabel_55.setText(_translate("sitool", "05C parameter 경로"))
        self.FitGroupConst_5.setTitle(_translate("sitool", "02C parameter"))
        self.aLabel_2.setText(_translate("sitool", "a"))
        self.bLabel_2.setText(_translate("sitool", "b"))
        self.b1Label_2.setText(_translate("sitool", "b1"))
        self.cLabel_2.setText(_translate("sitool", "c"))
        self.dLabel_2.setText(_translate("sitool", "d"))
        self.eLabel_2.setText(_translate("sitool", "e"))
        self.fLabel_2.setText(_translate("sitool", "f"))
        self.fdLabel_2.setText(_translate("sitool", "f_d"))
        self.FitGroupConst_6.setTitle(_translate("sitool", "05C parameter"))
        self.aLabel_8.setText(_translate("sitool", "a"))
        self.bLabel_8.setText(_translate("sitool", "b"))
        self.b1Label_8.setText(_translate("sitool", "b1"))
        self.cLabel_8.setText(_translate("sitool", "c"))
        self.dLabel_8.setText(_translate("sitool", "d"))
        self.eLabel_8.setText(_translate("sitool", "e"))
        self.fLabel_8.setText(_translate("sitool", "f"))
        self.fdLabel_8.setText(_translate("sitool", "f_d"))
        self.cyc_long_life.setText(_translate("sitool", "평가 중 장수명 적용"))
        self.simul_long_life.setText(_translate("sitool", "결과 중 장수명 반영"))
        self.cycxlabel_33.setText(_translate("sitool", "  Y축 최대"))
        self.simul_y_max.setText(_translate("sitool", "1.1"))
        self.cycxlabel_17.setText(_translate("sitool", "  Y축 최소"))
        self.simul_y_min.setText(_translate("sitool", "0.8"))
        self.cycxlabel_34.setText(_translate("sitool", "  X축 최대"))
        self.simul_x_max.setText(_translate("sitool", "2000"))
        self.load_cycparameter.setText(_translate("sitool", "Parameter Load"))
        self.AppCycleTabReset.setText(_translate("sitool", "Tab Reset"))
        self.pathappcycestimation.setText(_translate("sitool", "Cyclepath 선택"))
        self.folderappcycestimation.setText(_translate("sitool", "데이터 file 선택"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), _translate("sitool", "승인 수명 예측"))
        self.selectcycle.setTitle(_translate("sitool", "용량 저항 선택"))
        self.chkcapacity.setText(_translate("sitool", "용량"))
        self.chkdcir.setText(_translate("sitool", "저항"))
        self.selectcapacity.setTitle(_translate("sitool", "싸이클 저장 선택"))
        self.chkcycle.setText(_translate("sitool", "싸이클"))
        self.chkstorage.setText(_translate("sitool", "저장"))
        self.selectlonglifecyc.setTitle(_translate("sitool", "장수명 적용"))
        self.nolonglife.setText(_translate("sitool", "장수명 미적용"))
        self.hhp_longlife.setText(_translate("sitool", "장수명 적용"))
        self.SimulTabResetConfirm.setText(_translate("sitool", "Tab Reset"))
        self.SimulConfirm.setText(_translate("sitool", "시뮬레이션"))
        self.label.setText(_translate("sitool", "장수명조건"))
        self.txt_longcycleno.setText(_translate("sitool", "0 300 400 700 1000"))
        self.txt_longcyclevol.setText(_translate("sitool", "0 0.02 0.04 0.06 0.11"))
        self.txt_relcap.setText(_translate("sitool", "96.5 95.1 93.7 92.3 88.8"))
        self.label_8.setText(_translate("sitool", "저장 Count1 (hr/cycle) (<=32도)"))
        self.txt_storageratio.setText(_translate("sitool", "0"))
        self.label_16.setText(_translate("sitool", "저장 Count2 (hr/cycle) (>32도)"))
        self.txt_storageratio2.setText(_translate("sitool", "0"))
        self.cycxlabel_53.setText(_translate("sitool", "Parameter 경로"))
        self.chk_cell_cycle.setText(_translate("sitool", "Cell수명"))
        self.chk_set_cycle.setText(_translate("sitool", "Set수명"))
        self.chk_detail_cycle.setText(_translate("sitool", "상세수명"))
        self.FitGroupConst.setTitle(_translate("sitool", "사이클 용량"))
        self.aLabel.setText(_translate("sitool", "a"))
        self.bLabel.setText(_translate("sitool", "b"))
        self.b1Label.setText(_translate("sitool", "b1"))
        self.cLabel.setText(_translate("sitool", "c"))
        self.dLabel.setText(_translate("sitool", "d"))
        self.eLabel.setText(_translate("sitool", "e"))
        self.fLabel.setText(_translate("sitool", "f"))
        self.fdLabel.setText(_translate("sitool", "f_d"))
        self.FitGroupConst_3.setTitle(_translate("sitool", "사이클 저항"))
        self.aLabel_5.setText(_translate("sitool", "a"))
        self.bLabel_5.setText(_translate("sitool", "b"))
        self.b1Label_5.setText(_translate("sitool", "b1"))
        self.cLabel_5.setText(_translate("sitool", "c"))
        self.dLabel_5.setText(_translate("sitool", "d"))
        self.eLabel_5.setText(_translate("sitool", "e"))
        self.fLabel_5.setText(_translate("sitool", "f"))
        self.fdLabel_5.setText(_translate("sitool", "f_d"))
        self.FitGroupConst_2.setTitle(_translate("sitool", "저장 용량"))
        self.aLabel_6.setText(_translate("sitool", "a"))
        self.bLabel_6.setText(_translate("sitool", "b"))
        self.b1Label_6.setText(_translate("sitool", "b1"))
        self.cLabel_6.setText(_translate("sitool", "c"))
        self.dLabel_6.setText(_translate("sitool", "d"))
        self.eLabel_6.setText(_translate("sitool", "e"))
        self.fLabel_6.setText(_translate("sitool", "f"))
        self.fdLabel_6.setText(_translate("sitool", "f_d"))
        self.FitGroupConst_4.setTitle(_translate("sitool", "저장 저항"))
        self.aLabel_7.setText(_translate("sitool", "a"))
        self.bLabel_7.setText(_translate("sitool", "b"))
        self.b1Label_7.setText(_translate("sitool", "b1"))
        self.cLabel_7.setText(_translate("sitool", "c"))
        self.dLabel_7.setText(_translate("sitool", "d"))
        self.eLabel_7.setText(_translate("sitool", "e"))
        self.fLabel_7.setText(_translate("sitool", "f"))
        self.fdLabel_7.setText(_translate("sitool", "f_d"))
        self.SimGroupConst.setTitle(_translate("sitool", "시뮬레이션"))
        self.xaxixLabel.setText(_translate("sitool", "X축 max"))
        self.xaxixTextEdit.setText(_translate("sitool", "1500"))
        self.xaxixLabel_2.setText(_translate("sitool", "사용량"))
        self.UsedCapTextEdit.setText(_translate("sitool", "1"))
        self.DODLabel.setText(_translate("sitool", "DOD (비율)"))
        self.DODTextEdit.setText(_translate("sitool", "1"))
        self.CrateLabel.setText(_translate("sitool", "충전Crate"))
        self.CrateTextEdit.setText(_translate("sitool", "1"))
        self.SOCLabel.setText(_translate("sitool", "최대전압(V)"))
        self.SOCTextEdit.setText(_translate("sitool", "4.43"))
        self.CrateLabel_2.setText(_translate("sitool", "방전Crate"))
        self.DcrateTextEdit.setText(_translate("sitool", "1"))
        self.TempLabel.setText(_translate("sitool", "온도(℃)"))
        self.TempTextEdit.setText(_translate("sitool", "23"))
        self.SOCLabel_3.setText(_translate("sitool", "저장전압 (V)"))
        self.SOCTextEdit_3.setText(_translate("sitool", "4.43"))
        self.TempLabel_3.setText(_translate("sitool", "온도(℃)"))
        self.TempTextEdit_3.setText(_translate("sitool", "23"))
        self.RestLabel_2.setText(_translate("sitool", "방치시간(day)"))
        self.RestTextEdit_2.setText(_translate("sitool", "0.167"))
        self.SOCLabel_2.setText(_translate("sitool", "저장전압 (V)"))
        self.SOCTextEdit_2.setText(_translate("sitool", "4.43"))
        self.TempLabel_2.setText(_translate("sitool", "온도(℃)"))
        self.TempTextEdit_2.setText(_translate("sitool", "23"))
        self.RestLabel.setText(_translate("sitool", "방치시간(day)"))
        self.RestTextEdit.setText(_translate("sitool", "0.167"))
        self.FdLabel.setText(_translate("sitool", "수명용량"))
        self.FdLabel_2.setText(_translate("sitool", "수명저항"))
        self.FdLabel_5.setText(_translate("sitool", "저장1용량"))
        self.FdLabel_6.setText(_translate("sitool", "저장1저항"))
        self.FdTextEdit_6.setText(_translate("sitool", "1"))
        self.FdTextEdit_3.setText(_translate("sitool", "1"))
        self.FdTextEdit.setText(_translate("sitool", "1"))
        self.FdLabel_3.setText(_translate("sitool", "저장2저항"))
        self.FdLabel_4.setText(_translate("sitool", "저장2용량"))
        self.FdTextEdit_5.setText(_translate("sitool", "1"))
        self.FdTextEdit_2.setText(_translate("sitool", "1"))
        self.FdTextEdit_4.setText(_translate("sitool", "1"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.FitTab), _translate("sitool", "실수명 예측"))
        self.mount_toyo.setText(_translate("sitool", "Z: 15F B Toyo"))
        self.mount_pne_1.setText(_translate("sitool", "Y: 15F B PNE1~2"))
        self.mount_pne_2.setText(_translate("sitool", "X: 15F B PNE3~5"))
        self.mount_pne_3.setText(_translate("sitool", "W: 3F B PNE1~8"))
        self.mount_pne_4.setText(_translate("sitool", "V: 3F B PNE9~16"))
        self.mount_pne_5.setText(_translate("sitool", "U: 3F A PNE17~21"))
        self.mount_all.setText(_translate("sitool", "All mount"))
        self.unmount_all.setText(_translate("sitool", "All unmount"))
        self.saveok.setText(_translate("sitool", "데이터 저장"))
        self.ect_saveok.setText(_translate("sitool", "ECT용 데이터 저장"))
        self.figsaveok.setText(_translate("sitool", "그림 저장"))

class WindowClass(QtWidgets.QMainWindow, Ui_sitool):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.chnlnow = "default"
        self.tab_no = 0
        # 충방전기 세팅 관련
        self.toyo_blk_list = ['BLK1', 'BLK2', 'BLK3', 'BLK4', 'BLK5']
        self.toyo_column_list = ['chno', 'use', 'testname', 'folder', 'day', 'part', 'name', 'temp', 'cyc', 'vol', 'path']
        self.toyo_cycler_name = ["Toyo #1", "Toyo #2", "Toyo #3", "Toyo #4", "Toyo #5"]
        self.pne_blk_list = ['PNE1', 'PNE2', 'PNE3', 'PNE4', 'PNE5', 'PNE01', 'PNE02', 'PNE03', 'PNE04', 'PNE05', 'PNE06',
                             'PNE07', 'PNE08', 'PNE09', 'PNE10', 'PNE11', 'PNE12', 'PNE13', 'PNE14', 'PNE15', 'PNE16',
                             'PNE17', 'PNE18', 'PNE19', 'PNE20', 'PNE21', 'PNE22', 'PNE23', 'PNE24', 'PNE25']
        self.pne_work_path_list = ['y:\\Working\\PNE1\\','y:\\Working\\PNE2\\',
                               'x:\\Working\\PNE3\\', 'x:\\Working\\PNE4\\', 'x:\\Working\\PNE5\\',
                               'w:\\Working\\PNE1\\', 'w:\\Working\\PNE2\\', 'w:\\Working\\PNE3\\', 'w:\\Working\\PNE4\\',
                               'w:\\Working\\PNE5\\', 'w:\\Working\\PNE6\\', 'w:\\Working\\PNE7\\', 'w:\\Working\\PNE8\\',
                               'v:\\Working\\PNE9\\', 'v:\\Working\\PNE10\\', 'v:\\Working\\PNE11\\', 'v:\\Working\\PNE12\\',
                               'v:\\Working\\PNE13\\', 'v:\\Working\\PNE14\\', 'v:\\Working\\PNE15\\', 'v:\\Working\\PNE16\\',
                               'u:\\Working\\PNE17\\', 'u:\\Working\\PNE18\\', 'u:\\Working\\PNE19\\', 'u:\\Working\\PNE20\\',
                               'u:\\Working\\PNE21\\', 'u:\\Working\\PNE22\\', 'u:\\Working\\PNE23\\', 'u:\\Working\\PNE24\\',
                               'u:\\Working\\PNE25\\']
        self.pne_data_path_list = ['y:\\PNE-1 New\\','y:\\PNE-2 New\\',
                               'x:\\PNE-3 New\\', 'x:\\PNE-4\\', 'x:\\PNE-5\\',
                               'w:\\PNE1\\', 'w:\\PNE2\\', 'w:\\PNE3\\', 'w:\\PNE4\\', 'w:\\PNE5\\',
                               'w:\\PNE6\\', 'w:\\PNE7\\', 'w:\\PNE8\\',
                               'v:\\PNE9\\', 'v:\\PNE10\\', 'v:\\PNE11\\', 'v:\\PNE12\\', 'v:\\PNE13\\',
                               'v:\\PNE14\\', 'v:\\PNE15\\', 'v:\\PNE16\\',
                               'u:\\PNE17\\', 'u:\\PNE18\\', 'u:\\PNE19\\', 'u:\\PNE20\\','u:\\PNE21\\',
                               'u:\\PNE22\\', 'u:\\PNE23\\', 'u:\\PNE24\\', 'u:\\PNE25\\']
        self.pne_cycler_name = ["PNE #1","PNE #2","PNE #3","PNE #4","PNE #5",
                            "PNE 3F - #01","PNE 3F - #02","PNE 3F - #03","PNE 3F - #04","PNE 3F - #05",
                            "PNE 3F - #06","PNE 3F - #07","PNE 3F - #08","PNE 3F - #09","PNE 3F - #10",
                            "PNE 3F - #11","PNE 3F - #12","PNE 3F - #13","PNE 3F - #14","PNE 3F - #15",
                            "PNE 3F - #16","PNE 3F - #17","PNE 3F - #18","PNE 3F - #19","PNE 3F - #20",
                            "PNE 3F - #21", "PNE 3F - #22", "PNE 3F - #23", "PNE 3F - #24", "PNE 3F - #25"]
        # 초기 폴더 경로 설정
        self.an_mat_dvdq_path.setText("d:/!dvdqraw/S25_291_anode_dchg_02C_gen4 수정.txt")
        self.ca_mat_dvdq_path.setText("d:/!dvdqraw/S25_291_cathode_dchg_02C.txt")
        self.pro_dvdq_path.setText("d:/!dvdqraw/Gen4 SDI 4512mAh/Gen4 SDI 450V 13C 4512mAh 45도 100CY.txt")
        self.cycparameter.setText("d:/!cyc_parameter_trend/para_data_Gen4 SDI MP1 업체 02C 수명 241212.txt")
        self.cycparameter2.setText("d:/!cyc_parameter_trend/para_data_Gen4 SDI MP1 업체 05C 수명 241212.txt")
        # UI 기준 초기치 설정 set-up
        self.firstCrate = float(self.ratetext.text())
        if self.inicaprate.isChecked():
            self.mincapacity = 0
        elif self.inicaptype.isChecked():
            self.mincapacity = float(self.capacitytext.text())
        self.xscale = int(self.tcyclerng.text())
        self.setxscale = int(self.setcyclexscale.text())
        self.ylimithigh = float(self.tcyclerngyhl.text())
        self.ylimitlow = float(self.tcyclerngyll.text())
        self.irscale = float(self.dcirscale.text())
        self.CycleNo = convert_steplist(self.stepnum.toPlainText())
        self.smoothdegree = int(self.smooth.text())
        self.mincrate = float(self.cutoff.text())
        self.dqscale = float(self.dqdvscale.text())
        self.dvscale = self.dqscale
        self.vol_y_hlimit = float(self.volrngyhl.text())
        self.vol_y_llimit = float(self.volrngyll.text())
        self.vol_y_gap = float(self.volrnggap.text())
        # 기초 dataframe 생성
        self.df = []
        self.AllchnlData = []
        self.ptn_df_select = []
        self.pne_ptn_merged_df = []
        # 각 버튼에 각각 명령어 할당
        # Combobox set up
        self.tb_info.currentIndexChanged.connect(self.tb_info_combobox)
        self.tb_cycler.currentIndexChanged.connect(self.tb_cycler_combobox)
        self.tb_room.currentIndexChanged.connect(self.tb_room_combobox)
        self.toyosumstate = 0
        self.pnesumstate = 0
        # unmount, mount button에 각각 명령어 할당
        self.mount_toyo.clicked.connect(self.mount_toyo_button)
        self.mount_pne_1.clicked.connect(self.mount_pne1_button)
        self.mount_pne_2.clicked.connect(self.mount_pne2_button)
        self.mount_pne_3.clicked.connect(self.mount_pne3_button)
        self.mount_pne_4.clicked.connect(self.mount_pne4_button)
        self.mount_pne_5.clicked.connect(self.mount_pne5_button)
        self.mount_all.clicked.connect(self.mount_all_button)
        self.unmount_all.clicked.connect(self.unmount_all_button)
        # 충방전기 데이터 보는 버튼
        self.cycle_tab_reset.clicked.connect(self.cycle_tab_reset_confirm_button)
        self.indiv_cycle.clicked.connect(self.indiv_cyc_confirm_button)
        self.overall_cycle.clicked.connect(self.overall_cyc_confirm_button)
        self.link_cycle.clicked.connect(self.link_cyc_confirm_button)
        self.link_cycle_indiv.clicked.connect(self.link_cyc_indiv_confirm_button)
        self.link_cycle_overall.clicked.connect(self.link_cyc_overall_confirm_button)
        self.AppCycConfirm.clicked.connect(self.app_cyc_confirm_button)
        self.StepConfirm.clicked.connect(self.step_confirm_button)
        self.RateConfirm.clicked.connect(self.rate_confirm_button)
        self.ChgConfirm.clicked.connect(self.chg_confirm_button)
        self.DchgConfirm.clicked.connect(self.dchg_confirm_button)
        self.ContinueConfirm.clicked.connect(self.continue_confirm_button)
        self.DCIRConfirm.clicked.connect(self.dcir_confirm_button)
        # SET 관련 버튼
        # self.BMSetProfile.clicked.connect(self.BMSetProfilebutton)
        # self.BMSetCycle.clicked.connect(self.BMSetCyclebutton)
        self.SETTabReset.clicked.connect(self.set_tab_reset_button)
        self.SetlogConfirm.clicked.connect(self.set_log_confirm_button)
        # self.SetlogcycConfirm.clicked.connect(self.SetlogcycConfirmbutton)
        self.SetConfirm.clicked.connect(self.set_confirm_button)
        self.SetCycle.clicked.connect(self.set_cycle_button)
        self.ECTShort.clicked.connect(self.ect_short_button)
        self.ECTSOC.clicked.connect(self.ect_soc_button)
        self.ECTSetProfile.clicked.connect(self.ect_set_profile_button)
        self.ECTSetCycle.clicked.connect(self.ect_set_cycle_button)
        self.ECTSetlog.clicked.connect(self.ect_set_log_button)
        self.ECTSetlog2.clicked.connect(self.ect_set_log2_button)
        # EU 수명 예측 버튼
        self.ParameterReset_eu.clicked.connect(self.eu_parameter_reset_button)
        self.TabReset_eu.clicked.connect(self.eu_tab_reset_button)
        self.load_cycparameter_eu.clicked.connect(self.eu_load_cycparameter_button)
        self.save_cycparameter_eu.clicked.connect(self.eu_save_cycparameter_button)
        self.FitConfirm_eu.clicked.connect(self.eu_fitting_confirm_button)
        self.ConstFitConfirm_eu.clicked.connect(self.eu_constant_fitting_confirm_button)
        self.indivConstFitConfirm_eu.clicked.connect(self.eu_indiv_constant_fitting_confirm_button)
        # 승인 수명 예측 버튼
        self.load_cycparameter.clicked.connect(self.load_cycparameter_button)
        self.AppCycleTabReset.clicked.connect(self.app_cycle_tab_reset_button)
        self.folderappcycestimation.clicked.connect(self.folder_approval_cycle_estimation_button)
        self.pathappcycestimation.clicked.connect(self.path_approval_cycle_estimation_button)
        self.folderappcycestimation.setDisabled(True)
        self.pathappcycestimation.setDisabled(True)
        # 필드 수명 예측 관련 버튼
        self.SimulConfirm.clicked.connect(self.simulation_confirm_button)
        self.SimulTabResetConfirm.clicked.connect(self.simulation_tab_reset_confirm_button)
        # 패턴 수정 버튼
        self.chg_ptn.clicked.connect(self.ptn_change_pattern_button)
        self.chg_ptn_refi.clicked.connect(self.ptn_change_refi_button)
        self.chg_ptn_endi.clicked.connect(self.ptn_change_endi_button)
        self.chg_ptn_chgv.clicked.connect(self.ptn_change_chgv_button)
        self.chg_ptn_dchgv.clicked.connect(self.ptn_change_dchgv_button)
        self.chg_ptn_endv.clicked.connect(self.ptn_change_endv_button)
        self.chg_ptn_step.clicked.connect(self.ptn_change_step_button)
        self.ptn_load.clicked.connect(self.ptn_load_button)
        # dVdQ fitting
        self.min_rms = np.inf
        self.mat_dvdq_btn.clicked.connect(self.dvdq_material_button)
        self.pro_dvdq_btn.clicked.connect(self.dvdq_profile_button)
        self.dvdq_ini_reset.clicked.connect(self.dvdq_ini_reset_button)
        self.dvdq_fitting.clicked.connect(self.dvdq_fitting_button)
        self.dvdq_fitting_2.clicked.connect(self.dvdq_fitting2_button)
        self.fittingdegree = 1
        # cycle 초기 a변수 설정
        parini1 = [0.03, -18, 0.7, 2.3, -782, -0.28, 96, 1]
        # 저장 초기 변수 설정
        parini2 = [0.03, -18, 0.7, 2.3, -782, -0.28, 96, 1]
        # 초기 parameter 설정
        simul_parameter = [self.aTextEdit, self.bTextEdit, self.b1TextEdit, self.cTextEdit, self.dTextEdit, self.eTextEdit,
                           self.fTextEdit, self.fdTextEdit]
        for i, text_edit in enumerate(simul_parameter):
            text_edit.setText(str(parini1[i]))
            text_edit_2 = getattr(self, f"{text_edit.objectName()}_2")
            text_edit_2.setText(str(parini2[i]))
            text_edit_3 = getattr(self, f"{text_edit.objectName()}_3")
            text_edit_3.setText(str(parini1[i]))
            text_edit_4 = getattr(self, f"{text_edit.objectName()}_4")
            text_edit_4.setText(str(parini2[i]))
        if os.path.isdir("z:"):
            connect_change(self.mount_toyo)
        else:
            disconnect_change(self.mount_toyo)
        if os.path.isdir("y:"):
            connect_change(self.mount_pne_1)
        else:
            disconnect_change(self.mount_pne_1)
        if os.path.isdir("x:"):
            connect_change(self.mount_pne_2)
        else:
            disconnect_change(self.mount_pne_2)
        if os.path.isdir("w:"):
            connect_change(self.mount_pne_3)
        else:
            disconnect_change(self.mount_pne_3)
        if os.path.isdir("v:"):
            connect_change(self.mount_pne_4)
        else:
            disconnect_change(self.mount_pne_4)
        if os.path.isdir("u:"):
            connect_change(self.mount_pne_5)
        else:
            disconnect_change(self.mount_pne_5)

    def cyc_ini_set(self):
        # UI 기준 초기 설정 데이터
        firstCrate = float(self.ratetext.text())
        if self.inicaprate.isChecked():
            mincapacity = 0
        elif self.inicaptype.isChecked():
            mincapacity = float(self.capacitytext.text())
        xscale = int(self.tcyclerng.text())
        ylimithigh = float(self.tcyclerngyhl.text())
        ylimitlow = float(self.tcyclerngyll.text())
        irscale = float(self.dcirscale.text())
        return firstCrate, mincapacity, xscale, ylimithigh, ylimitlow, irscale

    def Profile_ini_set(self):
        # UI 기준 초기 설정 데이터
        firstCrate = float(self.ratetext.text())
        if self.inicaprate.isChecked():
            mincapacity = 0
        elif self.inicaptype.isChecked():
            mincapacity = float(self.capacitytext.text())
        CycleNo = convert_steplist(self.stepnum.toPlainText())
        smoothdegree = int(self.smooth.text())
        mincrate = float(self.cutoff.text())
        dqscale = float(self.dqdvscale.text())
        dvscale = dqscale
        self.vol_y_hlimit = float(self.volrngyhl.text())
        self.vol_y_llimit = float(self.volrngyll.text())
        self.vol_y_gap = float(self.volrnggap.text())
        return firstCrate, mincapacity, CycleNo, smoothdegree, mincrate, dqscale, dvscale

    def tab_delete(self, tab):
        while tab.count() > 0:
            tab.removeTab(0)

    #종료이벤트 발생시 종료
    def closeEvent(self, QCloseEvent):
        sys.exit()

    def inicaprate_on(self):
        self.inicaprate.setChecked(True)

    def inicaptype_on(self):
        self.inicaptype.setChecked(True)

    def pne_path_setting(self):
        all_data_name = []
        all_data_folder = []
        datafilepath = []
        # path file이 있을 경우
        if self.chk_cyclepath.isChecked():
            datafilepath = filedialog.askopenfilename(initialdir="d://", title="Choose Test files")
            if datafilepath:
                cycle_path = pd.read_csv(datafilepath, sep="\t", engine="c", encoding="UTF-8", skiprows=1, on_bad_lines='skip')
                if hasattr(cycle_path,"cyclepath"):
                    all_data_folder = np.array(cycle_path.cyclepath.tolist())
                    if hasattr(cycle_path,"cyclename"):
                        all_data_name = np.array(cycle_path.cyclename.tolist())
                    if (self.inicaprate.isChecked()) and ("mAh" in datafilepath):
                        self.mincapacity = name_capacity(datafilepath)
                        self.capacitytext.setText(str(self.mincapacity))
                else:
                    all_data_folder = multi_askopendirnames()
            else:
                all_data_folder = multi_askopendirnames()
        elif self.stepnum_2.toPlainText() != "":
            datafilepath = list(map(str, self.stepnum_2.toPlainText().split('\n')))
            all_data_folder = np.array(datafilepath)
        else:
            all_data_folder = multi_askopendirnames()
            datafilepath = all_data_folder
        return [all_data_folder, all_data_name, datafilepath]

    def app_pne_path_setting(self):
        all_data_name = []
        all_data_folder = multi_askopendirnames()
        return [all_data_folder, all_data_name]

    def cycle_tab_reset_confirm_button(self):
        self.tab_delete(self.cycle_tab)
        self.tab_no = 0
    
    def app_cyc_confirm_button(self):
        # 버튼 비활성화
        global writer
        self.AppCycState = True
        self.AppCycConfirm.setDisabled(True)
        firstCrate, mincapacity, xscale, ylimithigh, ylimitlow, irscale = self.cyc_ini_set()
        graphcolor = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
                      '#17becf', ]
        filecount,colorno , columncount = 0, 0, 0
        dfoutput = pd.DataFrame()
        col_name_output = []
        root = Tk()
        root.withdraw()
        filename = ""
        all_data_folder = filedialog.askopenfilenames(initialdir="d://", title="Choose Test files")
        if self.saveok.isChecked():
            save_file_name = filedialog.asksaveasfilename(initialdir="D://", title="Save File Name", defaultextension=".xlsx")
            if save_file_name:
                writer = pd.ExcelWriter(save_file_name, engine="xlsxwriter")
        self.AppCycConfirm.setEnabled(True)
        fig, ((ax1)) = plt.subplots(nrows=1, ncols=1, figsize=(14, 8))
        tab_no = 0
        for i, datafilepath in enumerate(all_data_folder):
            # tab 그래프 추가
            tab = QtWidgets.QWidget()
            tab_layout = QtWidgets.QVBoxLayout(tab)
            canvas = FigureCanvas(fig)
            toolbar = NavigationToolbar(canvas, None)
            if (self.inicaprate.isChecked()) and ("mAh" in datafilepath):
                mincapacity = name_capacity(datafilepath)
                self.capacitytext.setText(str(mincapacity))
            else:
                mincapacity = float(self.capacitytext.text())
            filename = datafilepath.split(".x")[-2].split("/")[-1].split("\\")[-1]
            try:
                wb = xw.Book(datafilepath)
                df = wb.sheets("Plot Base Data").used_range.offset(1,0).options(pd.DataFrame, index=False, header=False).value
                xw.apps.active.quit()
                df = df.drop(0)
                df = df.iloc[:,1::2]
                # df = df.dropna(axis=0)
                df.reset_index(drop=True, inplace=True)
                df.index = df.index + 1
                col_name = [filename for i in range(0, len(df.columns))]
                df = df/mincapacity
                # df.index = df.index + 1
                if df.iat[2, 0] < df.iat[0, 0] * 0.5:
                    count = len(df)
                    lastcount = int((count + int(count / 199) + 1) / 2 + 1)
                    index = 0
                    for i in range(lastcount - 1):
                        if (index == 0) or (index == 197):
                            index = index + 1
                        else:
                            if (index > 197) and ((index - 197) % 199 == 0):
                                index = index + 1
                            else:
                                df.loc[index + 1,:] = df.loc[index + 1,:] + df.loc[index + 2,:]
                                df.drop(index + 2, axis=0, inplace=True)
                                index = index + 2
                    df.reset_index(drop=True, inplace=True)
                    df.index = df.index + 1
                columncount = 0
                for col, column in df.items():
                    if columncount == 0:
                        graph_cycle(df.index, column, ax1, ylimitlow, ylimithigh, 0.05, "Cycle", "Discharge Capacity Ratio",
                                    filename, xscale, graphcolor[colorno])
                    else:
                        graph_cycle(df.index, column, ax1, ylimitlow, ylimithigh, 0.05, "Cycle", "Discharge Capacity Ratio",
                                    "" , xscale, graphcolor[colorno])
                    columncount = columncount + 1
                    colorno = (colorno + 1)%10
                filecountmax = len(all_data_folder)
                progressdata = filecount/filecountmax * 100
                filecount = filecount + 1
                self.progressBar.setValue(int(progressdata))
                dfoutput = pd.concat([dfoutput, df], axis=1)
                col_name_output = col_name_output + col_name
            except Exception as e:
                print(f"오류 발생: {e}")
                raise
        if self.saveok.isChecked() and save_file_name:
            dfoutput.to_excel(writer, sheet_name="Approval_cycle", header = col_name_output)
            writer.close()
        if filename != "":
            plt.suptitle(filename, fontsize= 15, fontweight='bold')
            plt.legend(loc="upper right")
            plt.tight_layout(pad=1, w_pad=1, h_pad=1)
            self.progressBar.setValue(100)
            output_fig(self.figsaveok, filename)
            tab_layout.addWidget(toolbar)
            tab_layout.addWidget(canvas)
            self.cycle_tab.addTab(tab, str(tab_no))
            self.cycle_tab.setCurrentWidget(tab)
            tab_no = tab_no + 1
        plt.tight_layout(pad=1, w_pad=1, h_pad=1)
        plt.close()

    def indiv_cyc_confirm_button(self):
        firstCrate, mincapacity, xscale, ylimithigh, ylimitlow, irscale = self.cyc_ini_set()
        # 용량 선정 관련
        global writer
        foldercount, chnlcount, writecolno, writerowno, Chnl_num, colorno = 0, 0, 0, 0, 0, 0
        root = Tk()
        root.withdraw()
        self.indiv_cycle.setDisabled(True)
        pne_path = self.pne_path_setting()
        all_data_folder = pne_path[0]
        all_data_name = pne_path[1]
        if pne_path[2]:
            mincapacity = name_capacity(pne_path[2])
        if self.saveok.isChecked():
            save_file_name = filedialog.asksaveasfilename(initialdir="D://", title="Save File Name", defaultextension=".xlsx")
            if save_file_name:
                writer = pd.ExcelWriter(save_file_name, engine="xlsxwriter")
        self.indiv_cycle.setEnabled(True)
        graphcolor = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        j = 0
        # while self.cycle_tab.count() > 0:
        #     self.cycle_tab.removeTab(0)
        tab_no = 0
        for i, cyclefolder in enumerate(all_data_folder):
            fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3, figsize=(14, 8))
            if os.path.exists(cyclefolder):
                subfolder = [f.path for f in os.scandir(cyclefolder) if f.is_dir()]
                foldercountmax = len(all_data_folder)
                foldercount = foldercount + 1
                for FolderBase in subfolder:
                    # tab 그래프 추가
                    tab = QtWidgets.QWidget()
                    tab_layout = QtWidgets.QVBoxLayout(tab)
                    canvas = FigureCanvas(fig)
                    toolbar = NavigationToolbar(canvas, None)
                    chnlcountmax = len(subfolder)
                    chnlcount = chnlcount + 1
                    progressdata = progress(foldercount, foldercountmax, chnlcount, chnlcountmax, 1, 1)
                    self.progressBar.setValue(int(progressdata))
                    cycnamelist = FolderBase.split("\\")
                    headername = [cycnamelist[-2] + ", " + cycnamelist[-1]]
                    if len(all_data_name) != 0 and j == i:
                        lgnd = all_data_name[i]
                        j = j + 1
                    elif len(all_data_name) != 0 and j != i:
                        lgnd = ""
                    else:
                        lgnd = extract_text_in_brackets(cycnamelist[-1])
                    if not check_cycler(cyclefolder):
                        cyctemp = toyo_cycle_data(FolderBase, mincapacity, firstCrate, self.dcirchk_2.isChecked())
                    else:
                        cyctemp = pne_cycle_data(FolderBase, mincapacity, firstCrate, self.dcirchk.isChecked(),
                                                    self.dcirchk_2.isChecked(), self.mkdcir.isChecked())
                    if hasattr(cyctemp[1], "NewData"):
                        self.capacitytext.setText(str(cyctemp[0]))
                        irscale = float(self.dcirscale.text())
                        if irscale == 0 and cyctemp[0] != 0:
                            irscale = int(1/(cyctemp[0]/5000) + 1)//2 * 2
                        if self.mkdcir.isChecked() and hasattr(cyctemp[1].NewData, "dcir2"):
                            graph_output_cycle(cyctemp[1], xscale, ylimitlow, ylimithigh, irscale, lgnd, lgnd, colorno,
                                                graphcolor, self.mkdcir, ax1, ax2, ax3, ax4, ax5, ax6)
                        else:
                            graph_output_cycle(cyctemp[1], xscale, ylimitlow, ylimithigh, irscale, lgnd, lgnd, colorno,
                                                graphcolor, self.mkdcir, ax1, ax2, ax3, ax4, ax5, ax6)
                        colorno = colorno + 1
                        # # Data output option
                        if self.saveok.isChecked() and save_file_name:
                            output_data(cyctemp[1].NewData, "방전용량", writecolno, 0, "Dchg", headername)
                            output_data(cyctemp[1].NewData, "Rest End", writecolno, 0, "RndV", headername)
                            output_data(cyctemp[1].NewData, "평균 전압", writecolno, 0, "AvgV", headername)
                            output_data(cyctemp[1].NewData, "충방효율", writecolno, 0, "Eff", headername)
                            output_data(cyctemp[1].NewData, "충전용량", writecolno, 0, "Chg", headername)
                            output_data(cyctemp[1].NewData, "방충효율", writecolno, 0, "Eff2", headername)
                            output_data(cyctemp[1].NewData, "방전Energy", writecolno, 0, "DchgEng", headername)
                            cyctempdcir = cyctemp[1].NewData.dcir.dropna(axis=0)
                            if self.mkdcir.isChecked() and hasattr(cyctemp[1].NewData, "dcir2"):
                                cyctempdcir2 = cyctemp[1].NewData.dcir2.dropna(axis=0)
                                cyctemprssocv = cyctemp[1].NewData.rssocv.dropna(axis=0)
                                cyctemprssccv = cyctemp[1].NewData.rssccv.dropna(axis=0)
                                cyctempsoc70dcir = cyctemp[1].NewData.soc70_dcir.dropna(axis=0)
                                cyctempsoc70rssdcir = cyctemp[1].NewData.soc70_rss_dcir.dropna(axis=0)
                                output_data(cyctempsoc70dcir, "SOC70_DCIR", writecolno, 0, "soc70_dcir", headername)
                                output_data(cyctempsoc70rssdcir, "SOC70_RSS", writecolno, 0, "soc70_rss_dcir", headername)
                                output_data(cyctempdcir, "RSS", writecolno, 0, "dcir", headername)
                                output_data(cyctempdcir2, "DCIR", writecolno, 0, "dcir2", headername)
                                output_data(cyctempdcir, "RSS", writecolno, 0, "dcir", headername)
                                output_data(cyctemprssocv, "RSS_OCV", writecolno, 0, "rssocv", headername)
                                output_data(cyctemprssccv, "RSS_CCV", writecolno, 0, "rssccv", headername)
                            else:
                                output_data(cyctempdcir, "DCIR", writecolno, 0, "dcir", headername)
                            output_data(cyctemp[1].NewData, "충방전기CY", writecolno, 0, "OriCyc", headername)
                            writecolno = writecolno + 1
                    # if len(all_data_name) != 0:
                    plt.suptitle(cycnamelist[-2], fontsize= 15, fontweight='bold')
                    ax1.legend(loc="lower left")
                    ax2.legend(loc="lower right")
                    ax3.legend(loc="upper right")
                    ax4.legend(loc="upper right")
                    ax5.legend(loc="upper right")
                    ax6.legend(loc="lower right")
                    # else:
                    #     plt.suptitle(cycnamelist[-2], fontsize= 15, fontweight='bold')
                    #     # plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
                    #     ax6.legend(loc="lower right")
                tab_layout.addWidget(toolbar)
                tab_layout.addWidget(canvas)
                self.cycle_tab.addTab(tab, str(tab_no))
                self.cycle_tab.setCurrentWidget(tab)
                tab_no = tab_no + 1
                plt.tight_layout(pad=1, w_pad=1, h_pad=1)
                output_fig(self.figsaveok, cycnamelist[-2])
                colorno = 0
        if self.saveok.isChecked() and save_file_name:
            writer.close()
        plt.tight_layout(pad=1, w_pad=1, h_pad=1)
        self.progressBar.setValue(100)
        plt.close()

    def overall_cyc_confirm_button(self):
        firstCrate, mincapacity, xscale, ylimithigh, ylimitlow, irscale = self.cyc_ini_set()
        # 용량 선정 관련
        global writer
        foldercount, chnlcount, writecolno, writerowno, Chnl_num = 0, 0, 0, 0, 0
        root = Tk()
        root.withdraw()
        self.overall_cycle.setDisabled(True)
        pne_path = self.pne_path_setting()
        all_data_folder = pne_path[0]
        all_data_name = pne_path[1]
        mincapacity = name_capacity(pne_path[2])
        if len(pne_path[2]) != 0:
            if ".t" in pne_path[2][0]:
                overall_filename = pne_path[2][0].split(".t")[-2].split("/")[-1]
        if self.saveok.isChecked():
            save_file_name = filedialog.asksaveasfilename(initialdir="D://", title="Save File Name", defaultextension=".xlsx")
            if save_file_name:
                writer = pd.ExcelWriter(save_file_name, engine="xlsxwriter")
        self.overall_cycle.setEnabled(True)
        graphcolor = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        # Cycle 관련 (그래프통합)
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3, figsize=(14, 8))
        writecolno, colorno, j, overall_xlimit = 0, 0, 0, 0
        tab_no = 0
        for i, cyclefolder in enumerate(all_data_folder):
            if os.path.isdir(cyclefolder):
                subfolder = [f.path for f in os.scandir(cyclefolder) if f.is_dir()]
                foldercountmax = len(all_data_folder)
                foldercount = foldercount + 1
                for FolderBase in subfolder:
                    tab = QtWidgets.QWidget()
                    tab_layout = QtWidgets.QVBoxLayout(tab)
                    canvas = FigureCanvas(fig)
                    toolbar = NavigationToolbar(canvas, None)
                    chnlcountmax = len(subfolder)
                    chnlcount = chnlcount + 1
                    progressdata = progress(foldercount, foldercountmax, chnlcount, chnlcountmax, 1, 1)
                    self.progressBar.setValue(int(progressdata))
                    cycnamelist = FolderBase.split("\\")
                    headername = [cycnamelist[-2] + ", " + cycnamelist[-1]]
                    # 중복없이 같은 LOT끼리에서만 legend 추가
                    if len(all_data_name) != 0 and j == i:
                        temp_lgnd = all_data_name[i]
                        j = j + 1
                    elif len(all_data_name) == 0 and j == i:
                        temp_lgnd = cycnamelist[-2].split('_')[-1]
                        j = j + 1
                    else:
                        temp_lgnd = ""
                    if not check_cycler(cyclefolder):
                        cyctemp = toyo_cycle_data(FolderBase, mincapacity, firstCrate, self.dcirchk_2.isChecked())
                    else:
                        cyctemp = pne_cycle_data(FolderBase, mincapacity, firstCrate, self.dcirchk.isChecked(),
                                                    self.dcirchk_2.isChecked(), self.mkdcir.isChecked())
                    if hasattr(cyctemp[1], "NewData"):
                        self.capacitytext.setText(str(cyctemp[0]))
                        if float(self.dcirscale.text()) == 0:
                            irscale_new = int(1/(cyctemp[0]/5000) + 1)//2 * 2
                            irscale = max(irscale, irscale_new)
                        if len(cyctemp[1].NewData.index) > overall_xlimit:
                            overall_xlimit = len(cyctemp[1].NewData.index)
                        if self.mkdcir.isChecked() and hasattr(cyctemp[1].NewData, "dcir2"):
                            graph_output_cycle(cyctemp[1], xscale, ylimitlow, ylimithigh, irscale, temp_lgnd, temp_lgnd,
                                                colorno, graphcolor, self.mkdcir, ax1, ax2, ax3, ax4, ax5, ax6)
                        else:
                            graph_output_cycle(cyctemp[1], xscale, ylimitlow, ylimithigh, irscale, temp_lgnd, temp_lgnd, colorno,
                                                graphcolor, self.mkdcir, ax1, ax2, ax3, ax4, ax5, ax6)
                        # # Data output option
                        if self.saveok.isChecked() and save_file_name:
                            output_data(cyctemp[1].NewData, "방전용량", writecolno, writerowno, "Dchg", headername)
                            output_data(cyctemp[1].NewData, "Rest End", writecolno, writerowno, "RndV", headername)
                            output_data(cyctemp[1].NewData, "평균 전압", writecolno, writerowno, "AvgV", headername)
                            output_data(cyctemp[1].NewData, "충방효율", writecolno, writerowno, "Eff", headername)
                            output_data(cyctemp[1].NewData, "충전용량", writecolno, writerowno, "Chg", headername)
                            output_data(cyctemp[1].NewData, "방충효율", writecolno, writerowno, "Eff2", headername)
                            output_data(cyctemp[1].NewData, "방전Energy", writecolno, writerowno, "DchgEng", headername)
                            cyctempdcir = cyctemp[1].NewData.dcir.dropna(axis=0)
                            if self.mkdcir.isChecked() and hasattr(cyctemp[1].NewData, "dcir2"):
                                cyctempdcir2 = cyctemp[1].NewData.dcir2.dropna(axis=0)
                                cyctemprssocv = cyctemp[1].NewData.rssocv.dropna(axis=0)
                                cyctemprssccv = cyctemp[1].NewData.rssccv.dropna(axis=0)
                                cyctempsoc70dcir = cyctemp[1].NewData.soc70_dcir.dropna(axis=0)
                                cyctempsoc70rssdcir = cyctemp[1].NewData.soc70_rss_dcir.dropna(axis=0)
                                output_data(cyctempsoc70dcir, "SOC70_DCIR", writecolno, 0, "soc70_dcir", headername)
                                output_data(cyctempsoc70rssdcir, "SOC70_RSS", writecolno, 0, "soc70_rss_dcir", headername)
                                output_data(cyctempdcir, "RSS", writecolno, 0, "dcir", headername)
                                output_data(cyctempdcir2, "DCIR", writecolno, 0, "dcir2", headername)
                                output_data(cyctempdcir, "RSS", writecolno, 0, "dcir", headername)
                                output_data(cyctemprssocv, "RSS_OCV", writecolno, 0, "rssocv", headername)
                                output_data(cyctemprssccv, "RSS_CCV", writecolno, 0, "rssccv", headername)
                            else:
                                output_data(cyctempdcir, "DCIR", writecolno, 0, "dcir", headername)
                            writecolno = writecolno + 1
                colorno = colorno % 9 + 1
        if len(all_data_name) != 0:
            ax1.legend(loc="lower left")
            ax2.legend(loc="lower right")
            ax3.legend(loc="upper right")
            ax4.legend(loc="upper right")
            ax5.legend(loc="upper right")
            ax6.legend(loc="lower right")
        else:
            ax6.legend(loc="lower right")
        if "overall_filename" in locals():
            if self.chk_cyclepath.isChecked():
                output_fig(self.figsaveok, overall_filename)
            else:
                output_fig(self.figsaveok, str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        if len(all_data_folder) != 0:
            tab_layout.addWidget(toolbar)
            tab_layout.addWidget(canvas)
            self.cycle_tab.addTab(tab, str(tab_no))
            self.cycle_tab.setCurrentWidget(tab)
            tab_no = tab_no + 1
        if self.saveok.isChecked() and save_file_name:
            writer.close()
        plt.tight_layout(pad=1, w_pad=1, h_pad=1)
        self.progressBar.setValue(100)
        plt.close()

    def link_cyc_confirm_button(self):
        firstCrate, mincapacity, xscale, ylimithigh, ylimitlow, irscale = self.cyc_ini_set()
        # 용량 선정 관련
        global writer
        foldercount, chnlcount, writecolno, writerowno, Chnl_num = 0, 0, 0, 0, 0
        CycleMax = [0, 0, 0, 0, 0]
        link_writerownum = [0, 0, 0, 0, 0]
        root = Tk()
        root.withdraw()
        self.link_cycle.setDisabled(True)
        pne_path = self.pne_path_setting()
        all_data_folder = pne_path[0]
        all_data_name = pne_path[1]
        mincapacity = name_capacity(pne_path[2])
        if self.saveok.isChecked():
            save_file_name = filedialog.asksaveasfilename(initialdir="D://", title="Save File Name", defaultextension=".xlsx")
            if save_file_name:
                writer = pd.ExcelWriter(save_file_name, engine="xlsxwriter")
        self.link_cycle.setEnabled(True)
        graphcolor = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        # Cycle 관련 (그래프 연결)
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3, figsize=(14, 8))
        writecolno ,colorno, j = 0, 0, 0
        # while self.cycle_tab.count() > 0:
        #     self.cycle_tab.removeTab(0)
        tab_no = 0
        for i, cyclefolder in enumerate(all_data_folder):
        # for cyclefolder in all_data_folder:
            if os.path.exists(cyclefolder):
                subfolder = [f.path for f in os.scandir(cyclefolder) if f.is_dir()]
                foldercountmax = len(all_data_folder)
                foldercount = foldercount + 1
                colorno, writecolno , Chnl_num = 0, 0, 0
                for FolderBase in subfolder:
                    tab = QtWidgets.QWidget()
                    tab_layout = QtWidgets.QVBoxLayout(tab)
                    canvas = FigureCanvas(fig)
                    toolbar = NavigationToolbar(canvas, None)
                    chnlcountmax = len(subfolder)
                    chnlcount = chnlcount + 1
                    # progressdata = (foldercount + chnlcount/chnlcountmax - 1)/foldercountmax * 100
                    progressdata = progress(foldercount, foldercountmax, chnlcount, chnlcountmax, 1, 1)
                    self.progressBar.setValue(int(progressdata))
                    cycnamelist = FolderBase.split("\\")
                    headername = [cycnamelist[-2] + ", " + cycnamelist[-1]]
                    if len(all_data_name) != 0 and j == i:
                        lgnd = all_data_name[i]
                        j = j + 1
                    elif len(all_data_name) != 0 and j != i:
                        lgnd = ""
                    else:
                        lgnd = cycnamelist[-1]
                    if not check_cycler(cyclefolder):
                        cyctemp = toyo_cycle_data(FolderBase, mincapacity, firstCrate, self.dcirchk_2.isChecked())
                    else:
                        cyctemp = pne_cycle_data(FolderBase, mincapacity, firstCrate, self.dcirchk.isChecked(),
                                                 self.dcirchk_2.isChecked(), self.mkdcir.isChecked())
                    if hasattr(cyctemp[1], "NewData") and (len(link_writerownum) > Chnl_num):
                        writerowno = link_writerownum[Chnl_num] + CycleMax[Chnl_num]
                        cyctemp[1].NewData.index = cyctemp[1].NewData.index + writerowno
                        if xscale == 0:
                            xscale = len(cyctemp[1].NewData) * (foldercountmax + 1)
                        self.capacitytext.setText(str(cyctemp[0]))
                        if irscale == 0:
                            irscale = int(1/(cyctemp[0]/5000) + 1)//2 * 2
                        if len(all_data_name) == 0:
                            temp_lgnd = ""
                        else:
                            temp_lgnd = lgnd
                        if self.mkdcir.isChecked() and hasattr(cyctemp[1].NewData, "dcir2"):
                            graph_output_cycle(cyctemp[1], xscale, ylimitlow, ylimithigh, irscale, lgnd, temp_lgnd, colorno,
                                               graphcolor, self.mkdcir, ax1, ax2, ax3, ax4, ax5, ax6)
                        else:
                            graph_output_cycle(cyctemp[1], xscale, ylimitlow, ylimithigh, irscale, lgnd, temp_lgnd, colorno,
                                                graphcolor, self.mkdcir, ax1, ax2, ax3, ax4, ax5, ax6)
                        # # Data output option
                        if self.saveok.isChecked() and save_file_name:
                            output_data(cyctemp[1].NewData, "방전용량", writecolno, writerowno, "Dchg", headername)
                            output_data(cyctemp[1].NewData, "Rest End", writecolno, writerowno, "RndV", headername)
                            output_data(cyctemp[1].NewData, "평균 전압", writecolno, writerowno, "AvgV", headername)
                            output_data(cyctemp[1].NewData, "충방효율", writecolno, writerowno, "Eff", headername)
                            output_data(cyctemp[1].NewData, "충전용량", writecolno, writerowno, "Chg", headername)
                            output_data(cyctemp[1].NewData, "방충효율", writecolno, writerowno, "Eff2", headername)
                            output_data(cyctemp[1].NewData, "방전Energy", writecolno, writerowno, "DchgEng", headername)
                            cyctempdcir = cyctemp[1].NewData.dcir.dropna(axis=0)
                            if self.mkdcir.isChecked() and hasattr(cyctemp[1].NewData, "dcir2"):
                                cyctempdcir2 = cyctemp[1].NewData.dcir2.dropna(axis=0)
                                cyctemprssocv = cyctemp[1].NewData.rssocv.dropna(axis=0)
                                cyctemprssccv = cyctemp[1].NewData.rssccv.dropna(axis=0)
                                output_data(cyctempdcir2, "DCIR", writecolno, 0, "dcir2", headername)
                                output_data(cyctempdcir, "RSS", writecolno, 0, "dcir", headername)
                                output_data(cyctemprssocv, "RSS_OCV", writecolno, 0, "rssocv", headername)
                                output_data(cyctemprssccv, "RSS_CCV", writecolno, 0, "rssccv", headername)
                            else:
                                output_data(cyctempdcir, "DCIR", writecolno, 0, "dcir", headername)
                        colorno = colorno + 1
                        writecolno = writecolno + 1
                        CycleMax[Chnl_num] = len(cyctemp[1].NewData)
                        link_writerownum[Chnl_num] = writerowno
                        Chnl_num = Chnl_num + 1
        if "cycnamelist" in locals():
            if len(all_data_name) != 0:
                plt.suptitle(cycnamelist[-2], fontsize= 15, fontweight='bold')
                ax1.legend(loc="lower left")
                ax2.legend(loc="lower right")
                ax3.legend(loc="upper right")
                ax4.legend(loc="upper right")
                ax5.legend(loc="upper right")
                ax6.legend(loc="lower right")
            else:
                plt.suptitle(cycnamelist[-2],fontsize= 15, fontweight='bold')
                plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        tab_layout.addWidget(toolbar)
        tab_layout.addWidget(canvas)
        self.cycle_tab.addTab(tab, str(tab_no))
        self.cycle_tab.setCurrentWidget(tab)
        tab_no = tab_no + 1
        if "cycnamelist" in locals():
            output_fig(self.figsaveok, cycnamelist[-2])
        if self.saveok.isChecked() and save_file_name:
            writer.close()
        plt.tight_layout(pad=1, w_pad=1, h_pad=1)
        self.progressBar.setValue(100)
        plt.close()

    def link_cyc_indiv_confirm_button(self):
        firstCrate, mincapacity, xscale, ylimithigh, ylimitlow, irscale = self.cyc_ini_set()
        # 용량 선정 관련
        global writer
        root = Tk()
        root.withdraw()
        self.link_cycle.setDisabled(True)
        all_data_name = []
        all_data_folder = []
        datafilepath = []
        alldatafilepath = filedialog.askopenfilenames(initialdir="d://", title="Choose Test files")
        if self.saveok.isChecked():
            save_file_name = filedialog.asksaveasfilename(initialdir="D://", title="Save File Name", defaultextension=".xlsx")
            if save_file_name:
                writer = pd.ExcelWriter(save_file_name, engine="xlsxwriter")
        self.link_cycle.setEnabled(True)
        graphcolor = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        # Cycle 관련 (그래프 연결)
        writecolno ,colorno, j, writecolnomax = 0, 0, 0, 0
        tab_no = 0
        for k, datafilepath in enumerate(alldatafilepath):
            fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3, figsize=(14, 8))
            folder_cnt, chnl_cnt, writerowno, Chnl_num = 0, 0, 0, 0
            writecolno = writecolnomax
            colorno, j = 0, 0
            CycleMax = [0, 0, 0, 0, 0]
            link_writerownum = [0, 0, 0, 0, 0]
            cycle_path = pd.read_csv(datafilepath, sep="\t", engine="c", encoding="UTF-8", skiprows=1, on_bad_lines='skip')
            all_data_folder = np.array(cycle_path.cyclepath.tolist())
            if hasattr(cycle_path,"cyclename"):
                all_data_name = np.array(cycle_path.cyclename.tolist())
            if (self.inicaprate.isChecked()) and ("mAh" in datafilepath):
                mincapacity = name_capacity(datafilepath)
                self.capacitytext.setText(str(self.mincapacity))
            for i, cyclefolder in enumerate(all_data_folder):
            # for cyclefolder in all_data_folder:
                if os.path.exists(cyclefolder):
                    subfolder = [f.path for f in os.scandir(cyclefolder) if f.is_dir()]
                    folder_cnt_max = len(all_data_folder)
                    folder_cnt = folder_cnt + 1
                    colorno, writecolno , Chnl_num = 0, 0, 0
                    for j, FolderBase in enumerate(subfolder):
                        tab = QtWidgets.QWidget()
                        tab_layout = QtWidgets.QVBoxLayout(tab)
                        canvas = FigureCanvas(fig)
                        toolbar = NavigationToolbar(canvas, None)
                        chnl_cnt_max = len(subfolder)
                        chnl_cnt = chnl_cnt + 1
                        filepath_max = len(alldatafilepath)
                        progressdata = progress(1, filepath_max, folder_cnt, folder_cnt_max, chnl_cnt, chnl_cnt_max)
                        self.progressBar.setValue(int(progressdata))
                        cycnamelist = FolderBase.split("\\")
                        headername = [cycnamelist[-2] + ", " + cycnamelist[-1]]
                        if len(all_data_name) != 0 and i == 0 and j == 0:
                            lgnd = all_data_name[i]
                        elif len(all_data_name) != 0:
                            lgnd = ""
                        else:
                            lgnd = cycnamelist[-1]
                        if not check_cycler(cyclefolder):
                            cyctemp = toyo_cycle_data(FolderBase, mincapacity, firstCrate, self.dcirchk_2.isChecked())
                        else:
                            cyctemp = pne_cycle_data(FolderBase, mincapacity, firstCrate, self.dcirchk.isChecked(),
                                                     self.dcirchk_2.isChecked(), self.mkdcir.isChecked())
                        if hasattr(cyctemp[1], "NewData") and (len(link_writerownum) > Chnl_num):
                            writerowno = link_writerownum[Chnl_num] + CycleMax[Chnl_num]
                            cyctemp[1].NewData.index = cyctemp[1].NewData.index + writerowno
                            if xscale == 0:
                                xscale = len(cyctemp[1].NewData) * (folder_cnt_max + 1)
                            self.capacitytext.setText(str(cyctemp[0]))
                            if irscale == 0:
                                irscale = int(1/(cyctemp[0]/5000) + 1)//2 * 2
                            if len(all_data_name) == 0:
                                temp_lgnd = ""
                            else:
                                temp_lgnd = lgnd
                            if self.mkdcir.isChecked() and hasattr(cyctemp[1].NewData, "dcir2"):
                                graph_output_cycle(cyctemp[1], xscale, ylimitlow, ylimithigh, irscale, lgnd, temp_lgnd, colorno,
                                                   graphcolor, self.mkdcir, ax1, ax2, ax3, ax4, ax5, ax6)
                            else:
                                graph_output_cycle(cyctemp[1], xscale, ylimitlow, ylimithigh, irscale, lgnd, temp_lgnd, colorno,
                                                    graphcolor, self.mkdcir, ax1, ax2, ax3, ax4, ax5, ax6)
                            # # Data output option
                            if self.saveok.isChecked() and save_file_name:
                                output_data(cyctemp[1].NewData, "방전용량", writecolno, writerowno, "Dchg", headername)
                                output_data(cyctemp[1].NewData, "Rest End", writecolno, writerowno, "RndV", headername)
                                output_data(cyctemp[1].NewData, "평균 전압", writecolno, writerowno, "AvgV", headername)
                                output_data(cyctemp[1].NewData, "충방효율", writecolno, writerowno, "Eff", headername)
                                output_data(cyctemp[1].NewData, "충전용량", writecolno, writerowno, "Chg", headername)
                                output_data(cyctemp[1].NewData, "방충효율", writecolno, writerowno, "Eff2", headername)
                                output_data(cyctemp[1].NewData, "방전Energy", writecolno, writerowno, "DchgEng", headername)
                                cyctempdcir = cyctemp[1].NewData.dcir.dropna(axis=0)
                                if self.mkdcir.isChecked() and hasattr(cyctemp[1].NewData, "dcir2"):
                                    cyctempdcir2 = cyctemp[1].NewData.dcir2.dropna(axis=0)
                                    cyctemprssocv = cyctemp[1].NewData.rssocv.dropna(axis=0)
                                    cyctemprssccv = cyctemp[1].NewData.rssccv.dropna(axis=0)
                                    output_data(cyctempdcir2, "DCIR", writecolno, 0, "dcir2", headername)
                                    output_data(cyctempdcir, "RSS", writecolno, 0, "dcir", headername)
                                    output_data(cyctemprssocv, "RSS_OCV", writecolno, 0, "rssocv", headername)
                                    output_data(cyctemprssccv, "RSS_CCV", writecolno, 0, "rssccv", headername)
                                else:
                                    output_data(cyctempdcir, "DCIR", writecolno, 0, "dcir", headername)
                                output_data(cyctemp[1].NewData, "충방전기CY", writecolno, 0, "OriCyc", headername)
                                writecolno = writecolno + 1
                            colorno = colorno + 1
                            CycleMax[Chnl_num] = len(cyctemp[1].NewData)
                            link_writerownum[Chnl_num] = writerowno
                            Chnl_num = Chnl_num + 1
                            writecolnomax = max(writecolno, writecolnomax)
            if "cycnamelist" in locals():
                if len(all_data_name) != 0:
                    plt.suptitle(cycnamelist[-2], fontsize= 15, fontweight='bold')
                    ax1.legend(loc="lower left")
                    ax2.legend(loc="lower right")
                    ax3.legend(loc="upper right")
                    ax4.legend(loc="upper right")
                    ax5.legend(loc="upper right")
                    ax6.legend(loc="lower right")
                else:
                    plt.suptitle(cycnamelist[-2],fontsize= 15, fontweight='bold')
                    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
            tab_layout.addWidget(toolbar)
            tab_layout.addWidget(canvas)
            self.cycle_tab.addTab(tab, str(tab_no))
            self.cycle_tab.setCurrentWidget(tab)
            tab_no = tab_no + 1
            plt.tight_layout(pad=1, w_pad=1, h_pad=1)
            if "cycnamelist" in locals():
                output_fig(self.figsaveok, cycnamelist[-2])
        if self.saveok.isChecked() and save_file_name:
            writer.close()
        plt.tight_layout(pad=1, w_pad=1, h_pad=1)
        self.progressBar.setValue(100)
        plt.close()

    def link_cyc_overall_confirm_button(self):
        firstCrate, mincapacity, xscale, ylimithigh, ylimitlow, irscale = self.cyc_ini_set()
        # 용량 선정 관련
        global writer
        root = Tk()
        root.withdraw()
        self.link_cycle.setDisabled(True)
        all_data_name = []
        all_data_folder = []
        datafilepath = []
        alldatafilepath = filedialog.askopenfilenames(initialdir="d://", title="Choose Test files")
        if self.saveok.isChecked():
            save_file_name = filedialog.asksaveasfilename(initialdir="D://", title="Save File Name", defaultextension=".xlsx")
            if save_file_name:
                writer = pd.ExcelWriter(save_file_name, engine="xlsxwriter")
        self.link_cycle.setEnabled(True)
        graphcolor = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        # Cycle 관련 (그래프 연결)
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3, figsize=(14, 8))
        writecolno ,colorno, j, maxcolor, writecolnomax = 0, 0, 0, 0, 0
        tab_no = 0
        for k, datafilepath in enumerate(alldatafilepath):
            folder_cnt, chnl_cnt, writerowno, Chnl_num = 0, 0, 0, 0
            writecolno = writecolnomax
            CycleMax = [0, 0, 0, 0, 0]
            link_writerownum = [0, 0, 0, 0, 0]
            cycle_path = pd.read_csv(datafilepath, sep="\t", engine="c", encoding="UTF-8", skiprows=1, on_bad_lines='skip')
            all_data_folder = np.array(cycle_path.cyclepath.tolist())
            if hasattr(cycle_path,"cyclename"):
                all_data_name = np.array(cycle_path.cyclename.tolist())
            if (self.inicaprate.isChecked()) and ("mAh" in datafilepath):
                mincapacity = name_capacity(datafilepath)
                self.capacitytext.setText(str(self.mincapacity))
            for i, cyclefolder in enumerate(all_data_folder):
            # for cyclefolder in all_data_folder:
                if os.path.exists(cyclefolder):
                    subfolder = [f.path for f in os.scandir(cyclefolder) if f.is_dir()]
                    folder_cnt_max = len(all_data_folder)
                    folder_cnt = folder_cnt + 1
                    colorno, writecolno , Chnl_num = maxcolor, 0, 0
                    for j, FolderBase in enumerate(subfolder):
                        tab = QtWidgets.QWidget()
                        tab_layout = QtWidgets.QVBoxLayout(tab)
                        canvas = FigureCanvas(fig)
                        toolbar = NavigationToolbar(canvas, None)
                        chnl_cnt_max = len(subfolder)
                        chnl_cnt = chnl_cnt + 1
                        filepath_max = len(alldatafilepath)
                        progressdata = progress(1, filepath_max, folder_cnt, folder_cnt_max, chnl_cnt, chnl_cnt_max)
                        self.progressBar.setValue(int(progressdata))
                        cycnamelist = FolderBase.split("\\")
                        headername = [cycnamelist[-2] + ", " + cycnamelist[-1]]
                        if len(all_data_name) != 0 and i == 0 and j == 0:
                            lgnd = all_data_name[i]
                        elif len(all_data_name) != 0:
                            lgnd = ""
                        else:
                            lgnd = cycnamelist[-1]
                        if not check_cycler(cyclefolder):
                            cyctemp = toyo_cycle_data(FolderBase, mincapacity, firstCrate, self.dcirchk_2.isChecked())
                        else:
                            cyctemp = pne_cycle_data(FolderBase, mincapacity, firstCrate, self.dcirchk.isChecked(),
                                                     self.dcirchk_2.isChecked(), self.mkdcir.isChecked())
                        if hasattr(cyctemp[1], "NewData") and (len(link_writerownum) > Chnl_num):
                            writerowno = link_writerownum[Chnl_num] + CycleMax[Chnl_num]
                            cyctemp[1].NewData.index = cyctemp[1].NewData.index + writerowno
                            if xscale == 0:
                                xscale = len(cyctemp[1].NewData) * (folder_cnt_max + 1)
                            self.capacitytext.setText(str(cyctemp[0]))
                            if irscale == 0:
                                irscale = int(1/(cyctemp[0]/5000) + 1)//2 * 2
                            if len(all_data_name) == 0:
                                temp_lgnd = ""
                            else:
                                temp_lgnd = lgnd
                            if self.mkdcir.isChecked() and hasattr(cyctemp[1].NewData, "dcir2"):
                                graph_output_cycle(cyctemp[1], xscale, ylimitlow, ylimithigh, irscale, lgnd, temp_lgnd, colorno,
                                                   graphcolor, self.mkdcir, ax1, ax2, ax3, ax4, ax5, ax6)
                            else:
                                graph_output_cycle(cyctemp[1], xscale, ylimitlow, ylimithigh, irscale, lgnd, temp_lgnd, colorno,
                                                    graphcolor, self.mkdcir, ax1, ax2, ax3, ax4, ax5, ax6)
                            # Data output option
                            if self.saveok.isChecked() and save_file_name:
                                output_data(cyctemp[1].NewData, "방전용량", writecolno, writerowno, "Dchg", headername)
                                output_data(cyctemp[1].NewData, "Rest End", writecolno, writerowno, "RndV", headername)
                                output_data(cyctemp[1].NewData, "평균 전압", writecolno, writerowno, "AvgV", headername)
                                output_data(cyctemp[1].NewData, "충방효율", writecolno, writerowno, "Eff", headername)
                                output_data(cyctemp[1].NewData, "충전용량", writecolno, writerowno, "Chg", headername)
                                output_data(cyctemp[1].NewData, "방충효율", writecolno, writerowno, "Eff2", headername)
                                output_data(cyctemp[1].NewData, "방전Energy", writecolno, writerowno, "DchgEng", headername)
                                cyctempdcir = cyctemp[1].NewData.dcir.dropna(axis=0)
                                if self.mkdcir.isChecked() and hasattr(cyctemp[1].NewData, "dcir2"):
                                    cyctempdcir2 = cyctemp[1].NewData.dcir2.dropna(axis=0)
                                    cyctemprssocv = cyctemp[1].NewData.rssocv.dropna(axis=0)
                                    cyctemprssccv = cyctemp[1].NewData.rssccv.dropna(axis=0)
                                    output_data(cyctempdcir2, "DCIR", writecolno, 0, "dcir2", headername)
                                    output_data(cyctempdcir, "RSS", writecolno, 0, "dcir", headername)
                                    output_data(cyctemprssocv, "RSS_OCV", writecolno, 0, "rssocv", headername)
                                    output_data(cyctemprssccv, "RSS_CCV", writecolno, 0, "rssccv", headername)
                                else:
                                    output_data(cyctempdcir, "DCIR", writecolno, 0, "dcir", headername)
                                output_data(cyctemp[1].NewData, "충방전기CY", writecolno, 0, "OriCyc", headername)
                                writecolno = writecolno + 1
                            CycleMax[Chnl_num] = len(cyctemp[1].NewData)
                            link_writerownum[Chnl_num] = writerowno
                            Chnl_num = Chnl_num + 1
                            writecolnomax = max(writecolno, writecolnomax)
                colorno = colorno + 1
            maxcolor = max(colorno, maxcolor)
            if "cycnamelist" in locals():
                if len(all_data_name) != 0:
                    plt.suptitle(cycnamelist[-2], fontsize= 15, fontweight='bold')
                    ax1.legend(loc="lower left")
                    ax2.legend(loc="lower right")
                    ax3.legend(loc="upper right")
                    ax4.legend(loc="upper right")
                    ax5.legend(loc="upper right")
                    ax6.legend(loc="lower right")
                else:
                    plt.suptitle(cycnamelist[-2],fontsize= 15, fontweight='bold')
                    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        tab_layout.addWidget(toolbar)
        tab_layout.addWidget(canvas)
        self.cycle_tab.addTab(tab, str(tab_no))
        self.cycle_tab.setCurrentWidget(tab)
        tab_no = tab_no + 1
        plt.tight_layout(pad=1, w_pad=1, h_pad=1)
        if "cycnamelist" in locals():
            output_fig(self.figsaveok, cycnamelist[-2])
        if self.saveok.isChecked() and save_file_name:
            writer.close()
        plt.tight_layout(pad=1, w_pad=1, h_pad=1)
        self.progressBar.setValue(100)
        plt.close()

    def step_confirm_button(self):
        self.StepConfirm.setDisabled(True)
        firstCrate, mincapacity, CycleNo, smoothdegree, mincrate, dqscale, dvscale = self.Profile_ini_set()
        # 용량 선정 관련
        global writer
        write_column_num, folder_count, chnlcount, cyccount = 0, 0, 0, 0
        root = Tk()
        root.withdraw()
        pne_path = self.pne_path_setting()
        all_data_folder = pne_path[0]
        all_data_name = pne_path[1]
        self.StepConfirm.setEnabled(True)
        if self.saveok.isChecked():
            save_file_name = filedialog.asksaveasfilename(initialdir="D://", title="Save File Name", defaultextension=".xlsx")
            if save_file_name:
                writer = pd.ExcelWriter(save_file_name, engine="xlsxwriter")
        if self.ect_saveok.isChecked():
            # save_file_name = filedialog.asksaveasfilename(initialdir="D://", title="Save File Name", defaultextension=".csv")
            save_file_name = filedialog.asksaveasfilename(initialdir="D://", title="Save File Name")
        tab_no = 0
        for i, cyclefolder in enumerate(all_data_folder):
            if os.path.isdir(cyclefolder):
                subfolder = [f.path for f in os.scandir(cyclefolder) if f.is_dir()]
                foldercountmax = len(all_data_folder)
                folder_count = folder_count + 1
                if self.CycProfile.isChecked():
                    for FolderBase in subfolder:
                        fig, ((step_ax1, step_ax2, step_ax3) ,(step_ax4, step_ax5, step_ax6)) = plt.subplots(
                            nrows=2, ncols=3, figsize=(14, 10))
                        tab = QtWidgets.QWidget()
                        tab_layout = QtWidgets.QVBoxLayout(tab)
                        canvas = FigureCanvas(fig)
                        toolbar = NavigationToolbar(canvas, None)
                        chnlcount = chnlcount + 1
                        chnlcountmax = len(subfolder)
                        if "Pattern" not in FolderBase:
                            for Step_CycNo in CycleNo:
                                cyccountmax = len(CycleNo)
                                cyccount = cyccount + 1
                                progressdata = progress(folder_count, foldercountmax, chnlcount, chnlcountmax, cyccount, cyccountmax)
                                self.progressBar.setValue(int(progressdata))
                                step_namelist = FolderBase.split("\\")
                                headername = step_namelist[-2] + ", " + step_namelist[-1] + ", " + str(Step_CycNo) + "cy, "
                                lgnd = "%04d" % Step_CycNo
                                if not check_cycler(cyclefolder):
                                    temp = toyo_step_Profile_data( FolderBase, Step_CycNo, mincapacity, mincrate, firstCrate)
                                else:
                                    temp = pne_step_Profile_data( FolderBase, Step_CycNo, mincapacity, mincrate, firstCrate)
                                if len(all_data_name) == 0:
                                    temp_lgnd = ""
                                else:
                                    temp_lgnd = all_data_name[i] +" "+lgnd
                                if hasattr(temp[1], "stepchg"):
                                    if len(temp[1].stepchg) > 2:
                                        self.capacitytext.setText(str(temp[0]))
                                        graph_step(temp[1].stepchg.TimeMin, temp[1].stepchg.Vol, step_ax1, self.vol_y_hlimit, self.vol_y_llimit,
                                                   self.vol_y_gap, "Time(min)", "Voltage(V)", temp_lgnd)
                                        graph_step(temp[1].stepchg.TimeMin, temp[1].stepchg.Vol, step_ax3, self.vol_y_hlimit, self.vol_y_llimit,
                                                   self.vol_y_gap, "Time(min)", "Voltage(V)", temp_lgnd)
                                        graph_step(temp[1].stepchg.TimeMin, temp[1].stepchg.Vol, step_ax2, self.vol_y_hlimit, self.vol_y_llimit,
                                                   self.vol_y_gap, "Time(min)", "Voltage(V)", temp_lgnd)
                                        graph_step(temp[1].stepchg.TimeMin, temp[1].stepchg.Crate, step_ax5, 0, 3.4, 0.2,
                                                   "Time(min)", "C-rate", temp_lgnd)
                                        graph_step(temp[1].stepchg.TimeMin, temp[1].stepchg.SOC, step_ax4, 0, 1.2, 0.1,
                                                   "Time(min)", "SOC", temp_lgnd)
                                        graph_step(temp[1].stepchg.TimeMin, temp[1].stepchg.Temp, step_ax6, -15, 60, 5,
                                                   "Time(min)", "Temperature (℃)", lgnd)
                                        # Data output option
                                        if self.saveok.isChecked() and save_file_name:
                                            temp[1].stepchg.to_excel(writer, startcol=write_column_num, index=False,
                                                                header=[headername + "time(min)",
                                                                        headername + "SOC",
                                                                        headername + "Voltage",
                                                                        headername + "Crate",
                                                                        headername + "Temp."])
                                            write_column_num = write_column_num + 5
                                        if self.ect_saveok.isChecked() and save_file_name:
                                            temp[1].stepchg["TimeSec"] = temp[1].stepchg.TimeMin * 60
                                            temp[1].stepchg["Curr"] = temp[1].stepchg.Crate * temp[0]/ 1000
                                            continue_df = temp[1].stepchg.loc[:,["TimeSec", "Vol", "Curr", "Temp"]]
                                            # 각 열을 소수점 자리수에 맞게 반올림
                                            continue_df['TimeSec'] = continue_df['TimeSec'].round(1)  # 소수점 1자리
                                            continue_df['Vol'] = continue_df['Vol'].round(4)           # 소수점 4자리
                                            continue_df['Curr'] = continue_df['Curr'].round(4)         # 소수점 4자리
                                            continue_df['Temp'] = continue_df['Temp'].round(1)         # 소수점 1자리
                                            continue_df.to_csv(save_file_name + "_" + "%04d" % Step_CycNo + ".csv", index=False, sep=',',
                                                                header=["time(s)",
                                                                        "Voltage(V)",
                                                                        "Current(A)",
                                                                        "Temp."])
                            title = step_namelist[-2] + "=" + step_namelist[-1]
                            plt.suptitle(title, fontsize= 15, fontweight='bold')
                            if len(all_data_name) != 0:
                                step_ax1.legend(loc="lower right")
                                step_ax2.legend(loc="lower right")
                                step_ax4.legend(loc="lower right")
                                step_ax3.legend(loc="lower right")
                                step_ax5.legend(loc="upper right")
                                step_ax6.legend(loc="upper right")
                            else:
                                plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
                            tab_layout.addWidget(toolbar)
                            tab_layout.addWidget(canvas)
                            self.cycle_tab.addTab(tab, str(tab_no))
                            self.cycle_tab.setCurrentWidget(tab)
                            tab_no = tab_no + 1
                            plt.tight_layout(pad=1, w_pad=1, h_pad=1)
                else:
                    for Step_CycNo in CycleNo:
                        fig, ((step_ax1, step_ax2, step_ax3) ,(step_ax4, step_ax5, step_ax6)) = plt.subplots(
                            nrows=2, ncols=3, figsize=(14, 10))
                        tab = QtWidgets.QWidget()
                        tab_layout = QtWidgets.QVBoxLayout(tab)
                        canvas = FigureCanvas(fig)
                        toolbar = NavigationToolbar(canvas, None)
                        chnlcount = chnlcount + 1
                        chnlcountmax = len(subfolder)
                        for FolderBase in subfolder:
                            if "Pattern" not in FolderBase:
                                cyccountmax = len(CycleNo)
                                cyccount = cyccount + 1
                                progressdata = progress(folder_count, foldercountmax, cyccount, cyccountmax, chnlcount, chnlcountmax)
                                self.progressBar.setValue(int(progressdata))
                                step_namelist = FolderBase.split("\\")
                                headername = step_namelist[-2] + ", " + step_namelist[-1] + ", " + str(Step_CycNo) + "cy, "
                                lgnd = step_namelist[-1]
                                if not check_cycler(cyclefolder):
                                    temp = toyo_step_Profile_data( FolderBase, Step_CycNo, mincapacity, mincrate, firstCrate)
                                else:
                                    temp = pne_step_Profile_data( FolderBase, Step_CycNo, mincapacity, mincrate, firstCrate)
                                if len(all_data_name) == 0:
                                    temp_lgnd = ""
                                else:
                                    temp_lgnd = all_data_name[i] +" "+lgnd
                                if hasattr(temp[1], "stepchg"):
                                    if len(temp[1].stepchg) > 2:
                                        self.capacitytext.setText(str(temp[0]))
                                        graph_step(temp[1].stepchg.TimeMin, temp[1].stepchg.Vol, step_ax1, self.vol_y_hlimit, self.vol_y_llimit, self.vol_y_gap,
                                                   "Time(min)", "Voltage(V)", temp_lgnd)
                                        graph_step(temp[1].stepchg.TimeMin, temp[1].stepchg.Vol, step_ax3, self.vol_y_hlimit, self.vol_y_llimit, self.vol_y_gap,
                                                   "Time(min)", "Voltage(V)", temp_lgnd)
                                        graph_step(temp[1].stepchg.TimeMin, temp[1].stepchg.Vol, step_ax2, self.vol_y_hlimit, self.vol_y_llimit, self.vol_y_gap,
                                                   "Time(min)", "Voltage(V)", temp_lgnd)
                                        graph_step(temp[1].stepchg.TimeMin, temp[1].stepchg.Crate, step_ax5, 0, 3.4, 0.2,
                                                   "Time(min)", "C-rate", temp_lgnd)
                                        graph_step(temp[1].stepchg.TimeMin, temp[1].stepchg.SOC, step_ax4, 0, 1.2, 0.1,
                                                   "Time(min)", "SOC", temp_lgnd)
                                        graph_step(temp[1].stepchg.TimeMin, temp[1].stepchg.Temp, step_ax6, -15, 60, 5,
                                                   "Time(min)", "Temperature (℃)", lgnd)
                                        # Data output option
                                        if self.saveok.isChecked() and save_file_name:
                                            temp[1].stepchg.to_excel(writer, startcol=write_column_num, index=False,
                                                                header=[headername + "time(min)",
                                                                        headername + "SOC",
                                                                        headername + "Voltage",
                                                                        headername + "Crate",
                                                                        headername + "Temp."])
                                            write_column_num = write_column_num + 5
                                title = step_namelist[-2] + "=" + "%04d" % Step_CycNo
                                plt.suptitle(title, fontsize= 15, fontweight='bold')
                                if len(all_data_name) != 0:
                                    step_ax1.legend(loc="lower right")
                                    step_ax2.legend(loc="lower right")
                                    step_ax4.legend(loc="lower right")
                                    step_ax3.legend(loc="lower right")
                                    step_ax5.legend(loc="upper right")
                                    step_ax6.legend(loc="upper right")
                                else:
                                    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
                        tab_layout.addWidget(toolbar)
                        tab_layout.addWidget(canvas)
                        self.cycle_tab.addTab(tab, str(tab_no))
                        self.cycle_tab.setCurrentWidget(tab)
                        tab_no = tab_no + 1
                        plt.tight_layout(pad=1, w_pad=1, h_pad=1)
        if self.saveok.isChecked() and save_file_name:
            writer.close()
        plt.tight_layout(pad=1, w_pad=1, h_pad=1)
        self.progressBar.setValue(100)
        plt.close()

    def rate_confirm_button(self):
        self.RateConfirm.setDisabled(True)
        firstCrate, mincapacity, CycleNo, smoothdegree, mincrate, dqscale, dvscale = self.Profile_ini_set()
        # 용량 선정 관련
        global writer
        writecolno, foldercount, chnlcount, cyccount = 0, 0, 0, 0
        root = Tk()
        root.withdraw()
        pne_path = self.pne_path_setting()
        all_data_folder = pne_path[0]
        all_data_name = pne_path[1]
        self.RateConfirm.setEnabled(True)
        if self.saveok.isChecked():
            save_file_name = filedialog.asksaveasfilename(initialdir="D://", title="Save File Name", defaultextension=".xlsx")
            if save_file_name:
                writer = pd.ExcelWriter(save_file_name, engine="xlsxwriter")
        if self.ect_saveok.isChecked():
            # save_file_name = filedialog.asksaveasfilename(initialdir="D://", title="Save File Name", defaultextension=".csv")
            save_file_name = filedialog.asksaveasfilename(initialdir="D://", title="Save File Name")
        tab_no = 0
        for i, cyclefolder in enumerate(all_data_folder):
            subfolder = [f.path for f in os.scandir(cyclefolder) if f.is_dir()]
            foldercountmax = len(all_data_folder)
            foldercount = foldercount + 1
            if self.CycProfile.isChecked():
                for FolderBase in subfolder:
                    fig, ((rate_ax1, rate_ax2, rate_ax3) ,(rate_ax4, rate_ax5, rate_ax6)) = plt.subplots(
                        nrows=2, ncols=3, figsize=(14, 10))
                    tab = QtWidgets.QWidget()
                    tab_layout = QtWidgets.QVBoxLayout(tab)
                    canvas = FigureCanvas(fig)
                    toolbar = NavigationToolbar(canvas, None)
                    chnlcount = chnlcount + 1
                    chnlcountmax = len(subfolder)
                    if "Pattern" not in FolderBase:
                        for CycNo in CycleNo:
                            cyccountmax = len(CycleNo)
                            cyccount = cyccount + 1
                            progressdata = progress(foldercount, foldercountmax, chnlcount, chnlcountmax, cyccount, cyccountmax)
                            self.progressBar.setValue(int(progressdata))
                            Ratenamelist = FolderBase.split("\\")
                            headername = Ratenamelist[-2] + ", " + Ratenamelist[-1] + ", " + str(CycNo) + "cy, "
                            lgnd = "%04d" % CycNo
                            if not check_cycler(cyclefolder):
                                Ratetemp = toyo_rate_Profile_data( FolderBase, CycNo, mincapacity, mincrate, firstCrate)
                            else:
                                Ratetemp = pne_rate_Profile_data( FolderBase, CycNo, mincapacity, mincrate, firstCrate)
                            if len(all_data_name) == 0:
                                temp_lgnd = ""
                            else:	
                                temp_lgnd = all_data_name[i] + " " + lgnd
                            if hasattr(Ratetemp[1], "rateProfile"):
                                if len(Ratetemp[1].rateProfile) > 2:
                                    self.capacitytext.setText(str(Ratetemp[0]))
                                    graph_step(Ratetemp[1].rateProfile.TimeMin, Ratetemp[1].rateProfile.Vol, rate_ax1, self.vol_y_hlimit, self.vol_y_llimit, self.vol_y_gap,
                                               "Time(min)", "Voltage(V)", temp_lgnd)
                                    graph_step(Ratetemp[1].rateProfile.TimeMin, Ratetemp[1].rateProfile.Vol, rate_ax4, self.vol_y_hlimit, self.vol_y_llimit, self.vol_y_gap,
                                               "Time(min)", "Voltage(V)", temp_lgnd)
                                    graph_step(Ratetemp[1].rateProfile.TimeMin, Ratetemp[1].rateProfile.Crate, rate_ax2, 0, 3.4, 0.2,
                                               "Time(min)", "C-rate", temp_lgnd)
                                    graph_step(Ratetemp[1].rateProfile.TimeMin, Ratetemp[1].rateProfile.Crate, rate_ax5, 0, 3.4, 0.2,
                                               "Time(min)", "C-rate", temp_lgnd)
                                    graph_step(Ratetemp[1].rateProfile.TimeMin, Ratetemp[1].rateProfile.SOC, rate_ax3, 0, 1.2, 0.1,
                                               "Time(min)", "SOC", temp_lgnd)
                                    graph_step(Ratetemp[1].rateProfile.TimeMin, Ratetemp[1].rateProfile.Temp, rate_ax6, -15, 60, 5,
                                               "Time(min)", "Temp.", lgnd)
                                    # Data output option
                                    if self.saveok.isChecked() and save_file_name:
                                        Ratetemp[1].rateProfile.to_excel(
                                            writer,
                                            startcol=writecolno,
                                            index=False,
                                            header=[
                                                headername + "time(min)",
                                                headername + "SOC",
                                                headername + "Voltage",
                                                headername + "Crate",
                                                headername + "Temp."
                                            ])
                                        writecolno = writecolno + 5
                                    if self.ect_saveok.isChecked() and save_file_name:
                                        Ratetemp[1].Profile["TimeSec"] = Ratetemp[1].Profile.TimeMin * 60
                                        Ratetemp[1].Profile["Curr"] = Ratetemp[1].Profile.Crate * Ratetemp[0] /1000
                                        continue_df = Ratetemp[1].Profile.loc[:,["TimeSec", "Vol", "Curr", "Temp"]]
                                        # 각 열을 소수점 자리수에 맞게 반올림
                                        continue_df['TimeSec'] = continue_df['TimeSec'].round(1)  # 소수점 1자리
                                        continue_df['Vol'] = continue_df['Vol'].round(4)           # 소수점 4자리
                                        continue_df['Curr'] = continue_df['Curr'].round(4)         # 소수점 4자리
                                        continue_df['Temp'] = continue_df['Temp'].round(1)         # 소수점 1자리
                                        continue_df.to_csv(save_file_name + "_" + "%04d" % CycNo + ".csv", index=False, sep=',',
                                                            header=["time(s)",
                                                                    "Voltage(V)",
                                                                    "Current(A)",
                                                                    "Temp."])
                            title = Ratenamelist[-2] + "=" + Ratenamelist[-1]
                            plt.suptitle(title, fontsize= 15, fontweight='bold')
                            if len(all_data_name) != 0:
                                rate_ax1.legend(loc="lower right")
                                rate_ax2.legend(loc="upper right")
                                rate_ax3.legend(loc="lower right")
                                rate_ax4.legend(loc="lower right")
                                rate_ax5.legend(loc="upper right")
                                rate_ax6.legend(loc="upper right")
                            else:
                                plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
                        tab_layout.addWidget(toolbar)
                        tab_layout.addWidget(canvas)
                        self.cycle_tab.addTab(tab, str(tab_no))
                        self.cycle_tab.setCurrentWidget(tab)
                        tab_no = tab_no + 1
                        plt.tight_layout(pad=1, w_pad=1, h_pad=1)
                        output_fig(self.figsaveok, title)
            else:
                for CycNo in CycleNo:
                    fig, ((rate_ax1, rate_ax2, rate_ax3) ,(rate_ax4, rate_ax5, rate_ax6)) = plt.subplots(
                        nrows=2, ncols=3, figsize=(14, 10))
                    tab = QtWidgets.QWidget()
                    tab_layout = QtWidgets.QVBoxLayout(tab)
                    canvas = FigureCanvas(fig)
                    toolbar = NavigationToolbar(canvas, None)
                    chnlcount = chnlcount + 1
                    chnlcountmax = len(subfolder)
                    for FolderBase in subfolder:
                        if "Pattern" not in FolderBase:
                            cyccountmax = len(CycleNo)
                            cyccount = cyccount + 1
                            progressdata = progress(foldercount, foldercountmax, cyccount, cyccountmax, chnlcount, chnlcountmax)
                            self.progressBar.setValue(int(progressdata))
                            Ratenamelist = FolderBase.split("\\")
                            headername = Ratenamelist[-2] + ", " + Ratenamelist[-1] + ", " + str(CycNo) + "cy, "
                            lgnd = Ratenamelist[-1]
                            if not check_cycler(cyclefolder):
                                Ratetemp = toyo_rate_Profile_data( FolderBase, CycNo, mincapacity, mincrate, firstCrate)
                            else:
                                Ratetemp = pne_rate_Profile_data( FolderBase, CycNo, mincapacity, mincrate, firstCrate)
                            if len(all_data_name) == 0:
                                temp_lgnd = ""
                            else:	
                                temp_lgnd = all_data_name[i] + " " + lgnd
                            if hasattr(Ratetemp[1], "rateProfile"):
                                if len(Ratetemp[1].rateProfile) > 2:
                                    self.capacitytext.setText(str(Ratetemp[0]))
                                    graph_step(Ratetemp[1].rateProfile.TimeMin, Ratetemp[1].rateProfile.Vol, rate_ax1, self.vol_y_hlimit,
                                               self.vol_y_llimit, self.vol_y_gap, "Time(min)", "Voltage(V)", temp_lgnd)
                                    graph_step(Ratetemp[1].rateProfile.TimeMin, Ratetemp[1].rateProfile.Vol, rate_ax4, self.vol_y_hlimit,
                                               self.vol_y_llimit, self.vol_y_gap, "Time(min)", "Voltage(V)", temp_lgnd)
                                    graph_step(Ratetemp[1].rateProfile.TimeMin, Ratetemp[1].rateProfile.Crate, rate_ax2, 0, 3.4, 0.2,
                                               "Time(min)", "C-rate", temp_lgnd)
                                    graph_step(Ratetemp[1].rateProfile.TimeMin, Ratetemp[1].rateProfile.Crate, rate_ax5, 0, 3.4, 0.2,
                                               "Time(min)", "C-rate", temp_lgnd)
                                    graph_step(Ratetemp[1].rateProfile.TimeMin, Ratetemp[1].rateProfile.SOC, rate_ax3, 0, 1.2, 0.1,
                                               "Time(min)", "SOC", temp_lgnd)
                                    graph_step(Ratetemp[1].rateProfile.TimeMin, Ratetemp[1].rateProfile.Temp, rate_ax6, -15, 60, 5,
                                               "Time(min)", "Temp.", lgnd)
                                    # Data output option
                                    if self.saveok.isChecked() and save_file_name:
                                        Ratetemp[1].rateProfile.to_excel(
                                            writer,
                                            startcol=writecolno,
                                            index=False,
                                            header=[
                                                headername + "time(min)",
                                                headername + "SOC",
                                                headername + "Voltage",
                                                headername + "Crate",
                                                headername + "Temp."
                                            ])
                                        writecolno = writecolno + 5
                            title = Ratenamelist[-2] + "=" + "%04d" % CycNo
                            plt.suptitle(title, fontsize= 15, fontweight='bold')
                            if len(all_data_name) != 0:
                                rate_ax1.legend(loc="lower right")
                                rate_ax2.legend(loc="upper right")
                                rate_ax3.legend(loc="lower right")
                                rate_ax4.legend(loc="lower right")
                                rate_ax5.legend(loc="upper right")
                                rate_ax6.legend(loc="upper right")
                            else:
                                plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
                    tab_layout.addWidget(toolbar)
                    tab_layout.addWidget(canvas)
                    self.cycle_tab.addTab(tab, str(tab_no))
                    self.cycle_tab.setCurrentWidget(tab)
                    tab_no = tab_no + 1
                    plt.tight_layout(pad=1, w_pad=1, h_pad=1)
                    output_fig(self.figsaveok, title)
        if self.saveok.isChecked() and save_file_name:
            writer.close()
        plt.tight_layout(pad=1, w_pad=1, h_pad=1)
        self.progressBar.setValue(100)
        plt.close()

    def chg_confirm_button(self):
        self.ChgConfirm.setDisabled(True)
        firstCrate, mincapacity, CycleNo, smoothdegree, mincrate, dqscale, dvscale = self.Profile_ini_set()
        # 용량 선정 관련
        global writer
        foldercount, chnlcount, cyccount, writecolno = 0, 0, 0, 0
        root = Tk()
        root.withdraw()
        pne_path = self.pne_path_setting()
        all_data_folder = pne_path[0]
        all_data_name = pne_path[1]
        self.ChgConfirm.setEnabled(True)
        if self.saveok.isChecked():
            save_file_name = filedialog.asksaveasfilename(initialdir="D://", title="Save File Name", defaultextension=".xlsx")
            if save_file_name:
                writer = pd.ExcelWriter(save_file_name, engine="xlsxwriter")
        if self.ect_saveok.isChecked():
            save_file_name = filedialog.asksaveasfilename(initialdir="D://", title="Save File Name")
        tab_no = 0
        for i, cyclefolder in enumerate(all_data_folder):
            if os.path.isdir(cyclefolder):
                subfolder = [f.path for f in os.scandir(cyclefolder) if f.is_dir()]
                foldercountmax = len(all_data_folder)
                foldercount = foldercount + 1
                if self.CycProfile.isChecked():
                    for FolderBase in subfolder:
                        chnlcount = chnlcount + 1
                        chnlcountmax = len(subfolder)
                        if "Pattern" not in FolderBase:
                            fig, ((Chg_ax1, Chg_ax2, Chg_ax3) ,(Chg_ax4, Chg_ax5, Chg_ax6)) = plt.subplots(
                                nrows=2, ncols=3, figsize=(14, 10))
                            tab = QtWidgets.QWidget()
                            tab_layout = QtWidgets.QVBoxLayout(tab)
                            canvas = FigureCanvas(fig)
                            toolbar = NavigationToolbar(canvas, None)
                            for CycNo in CycleNo:
                                cyccountmax = len(CycleNo)
                                cyccount = cyccount + 1
                                progressdata = progress(foldercount, foldercountmax, chnlcount, chnlcountmax, cyccount, cyccountmax)
                                self.progressBar.setValue(int(progressdata))
                                Chgnamelist = FolderBase.split("\\")
                                headername = Chgnamelist[-2] + ", " + Chgnamelist[-1] + ", " + str(CycNo) + "cy, "
                                lgnd = "%04d" % CycNo
                                if not check_cycler(cyclefolder):
                                    Chgtemp = toyo_chg_Profile_data( FolderBase, CycNo, mincapacity, mincrate, firstCrate,
                                                                    smoothdegree)
                                else:
                                    Chgtemp = pne_chg_Profile_data( FolderBase, CycNo, mincapacity, mincrate, firstCrate,
                                                                   smoothdegree)
                                if len(all_data_name) == 0:
                                    temp_lgnd = ""
                                else:	
                                    temp_lgnd = all_data_name[i] + " " + lgnd
                                if hasattr(Chgtemp[1], "Profile"):
                                    if len(Chgtemp[1].Profile) > 2:
                                        self.capacitytext.setText(str(Chgtemp[0]))
                                        graph_profile( Chgtemp[1].Profile.SOC, Chgtemp[1].Profile.Vol, Chg_ax1,
                                                      0, 1.3, 0.1, self.vol_y_hlimit, self.vol_y_llimit, self.vol_y_gap, "SOC", "Voltage(V)", temp_lgnd)
                                        graph_profile( Chgtemp[1].Profile.SOC, Chgtemp[1].Profile.Vol, Chg_ax3,
                                                      0, 1.3, 0.1, self.vol_y_hlimit, self.vol_y_llimit, self.vol_y_gap, "SOC", "Voltage(V)", temp_lgnd)
                                        if self.chk_dqdv.isChecked():
                                            graph_profile( Chgtemp[1].Profile.Vol, Chgtemp[1].Profile.dQdV, Chg_ax2,
                                                        self.vol_y_hlimit, self.vol_y_llimit, self.vol_y_gap, 0, 5.5 * dqscale, 0.5 * dqscale, 
                                                        "Voltage(V)","dQdV", temp_lgnd)
                                        else:
                                            graph_profile( Chgtemp[1].Profile.dQdV, Chgtemp[1].Profile.Vol, Chg_ax2,
                                                        0, 5.5 * dqscale, 0.5 * dqscale, self.vol_y_hlimit, self.vol_y_llimit, self.vol_y_gap,
                                                        "dQdV", "Voltage(V)", temp_lgnd)
                                        graph_profile( Chgtemp[1].Profile.SOC, Chgtemp[1].Profile.Crate, Chg_ax5,
                                                      0, 1.3, 0.1, 0, 3.4, 0.2, "SOC", "C-rate", temp_lgnd) 
                                        graph_profile( Chgtemp[1].Profile.SOC, Chgtemp[1].Profile.dVdQ, Chg_ax4,
                                                      0, 1.3, 0.1, 0, 5.5 * dvscale, 0.5 * dvscale, "SOC", "dVdQ", temp_lgnd)
                                        graph_profile( Chgtemp[1].Profile.SOC, Chgtemp[1].Profile.Temp, Chg_ax6,
                                                      0, 1.3, 0.1, -15, 60, 5, "SOC", "Temp.", lgnd)
                                        # Data output option
                                        if self.saveok.isChecked() and save_file_name:
                                            Chgtemp[1].Profile.to_excel(
                                                writer,
                                                startcol=writecolno,
                                                index=False,
                                                header=[
                                                    headername + "Time(min)",
                                                    headername + "SOC",
                                                    headername + "Energy",
                                                    headername + "Voltage",
                                                    headername + "Crate",
                                                    headername + "dQdV",
                                                    headername + "dVdQ",
                                                    headername + "Temp."
                                                ])
                                            writecolno = writecolno + 8
                                        if self.ect_saveok.isChecked() and save_file_name:
                                            Chgtemp[1].Profile["TimeSec"] = Chgtemp[1].Profile["TimeMin"] * 60
                                            Chgtemp[1].Profile["Curr"] = Chgtemp[1].Profile["Crate"] * Chgtemp[0] / 1000
                                            continue_df = Chgtemp[1].Profile.loc[:,["TimeSec", "Vol", "Curr", "Temp"]]
                                            # 각 열을 소수점 자리수에 맞게 반올림
                                            continue_df['TimeSec'] = continue_df['TimeSec'].round(1)  # 소수점 1자리
                                            continue_df['Vol'] = continue_df['Vol'].round(4)           # 소수점 4자리
                                            continue_df['Curr'] = continue_df['Curr'].round(4)         # 소수점 4자리
                                            continue_df['Temp'] = continue_df['Temp'].round(1)         # 소수점 1자리
                                            continue_df.to_csv(save_file_name + "_"+ "%04d" % CycNo + ".csv", index=False, sep=',',
                                                                header=["time(s)",
                                                                        "Voltage(V)",
                                                                        "Current(A)",
                                                                        "Temp."])
                                title = Chgnamelist[-2] + "=" + Chgnamelist[-1]
                                plt.suptitle(title, fontsize= 15, fontweight='bold')
                                if len(all_data_name) != 0:
                                    Chg_ax1.legend(loc="lower right")
                                    Chg_ax2.legend(loc="lower right")
                                    Chg_ax3.legend(loc="lower right")
                                    Chg_ax4.legend(loc="upper right")
                                    Chg_ax5.legend(loc="upper right")
                                    Chg_ax6.legend(loc="upper right")
                                else:
                                    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
                            tab_layout.addWidget(toolbar)
                            tab_layout.addWidget(canvas)
                            self.cycle_tab.addTab(tab, str(tab_no))
                            self.cycle_tab.setCurrentWidget(tab)
                            tab_no = tab_no + 1
                            plt.tight_layout(pad=1, w_pad=1, h_pad=1)
                            output_fig(self.figsaveok, title)
                else:
                    for CycNo in CycleNo:
                        chnlcount = chnlcount + 1
                        chnlcountmax = len(subfolder)
                        fig, ((Chg_ax1, Chg_ax2, Chg_ax3) ,(Chg_ax4, Chg_ax5, Chg_ax6)) = plt.subplots(
                            nrows=2, ncols=3, figsize=(14, 10))
                        tab = QtWidgets.QWidget()
                        tab_layout = QtWidgets.QVBoxLayout(tab)
                        canvas = FigureCanvas(fig)
                        toolbar = NavigationToolbar(canvas, None)
                        for FolderBase in subfolder:
                            if "Pattern" not in FolderBase:
                                cyccountmax = len(CycleNo)
                                cyccount = cyccount + 1
                                progressdata = progress(foldercount, foldercountmax, cyccount, cyccountmax, chnlcount, chnlcountmax)
                                self.progressBar.setValue(int(progressdata))
                                Chgnamelist = FolderBase.split("\\")
                                headername = Chgnamelist[-2] + ", " + Chgnamelist[-1] + ", " + str(CycNo) + "cy, "
                                lgnd = Chgnamelist[-1]
                                if not check_cycler(cyclefolder):
                                    Chgtemp = toyo_chg_Profile_data( FolderBase, CycNo, mincapacity, mincrate, firstCrate,
                                                                    smoothdegree)
                                else:
                                    Chgtemp = pne_chg_Profile_data( FolderBase, CycNo, mincapacity, mincrate, firstCrate,
                                                                   smoothdegree)
                                if len(all_data_name) == 0:
                                    temp_lgnd = ""
                                else:	
                                    temp_lgnd = all_data_name[i] + " " + lgnd
                                if hasattr(Chgtemp[1], "Profile"):
                                    if len(Chgtemp[1].Profile) > 2:
                                        self.capacitytext.setText(str(Chgtemp[0]))
                                        graph_profile( Chgtemp[1].Profile.SOC, Chgtemp[1].Profile.Vol, Chg_ax1,
                                                      0, 1.3, 0.1, self.vol_y_hlimit, self.vol_y_llimit, self.vol_y_gap, "SOC", "Voltage(V)", temp_lgnd)
                                        graph_profile( Chgtemp[1].Profile.SOC, Chgtemp[1].Profile.Vol, Chg_ax3,
                                                      0, 1.3, 0.1, self.vol_y_hlimit, self.vol_y_llimit, self.vol_y_gap, "SOC", "Voltage(V)", temp_lgnd)
                                        if self.chk_dqdv.isChecked():
                                            graph_profile( Chgtemp[1].Profile.Vol, Chgtemp[1].Profile.dQdV, Chg_ax2,
                                                        self.vol_y_hlimit, self.vol_y_llimit, self.vol_y_gap, 0, 5.5 * dqscale, 0.5 * dqscale, 
                                                        "Voltage(V)","dQdV", temp_lgnd)
                                        else:
                                            graph_profile( Chgtemp[1].Profile.dQdV, Chgtemp[1].Profile.Vol, Chg_ax2,
                                                        0, 5.5 * dqscale, 0.5 * dqscale, self.vol_y_hlimit, self.vol_y_llimit, self.vol_y_gap,
                                                        "dQdV", "Voltage(V)", temp_lgnd)
                                        graph_profile( Chgtemp[1].Profile.dQdV, Chgtemp[1].Profile.Vol, Chg_ax2, 0, 5.5 * dqscale, 0.5 * dqscale,
                                                      self.vol_y_hlimit, self.vol_y_llimit, self.vol_y_gap, "dQdV", "Voltage(V)", temp_lgnd)
                                        graph_profile( Chgtemp[1].Profile.SOC, Chgtemp[1].Profile.Crate, Chg_ax5,
                                                      0, 1.3, 0.1, 0, 3.4, 0.2, "SOC", "C-rate", temp_lgnd) 
                                        graph_profile( Chgtemp[1].Profile.SOC, Chgtemp[1].Profile.dVdQ, Chg_ax4,
                                                      0, 1.3, 0.1, 0, 5.5 * dvscale, 0.5 * dvscale, "SOC", "dVdQ", temp_lgnd)
                                        graph_profile( Chgtemp[1].Profile.SOC, Chgtemp[1].Profile.Temp, Chg_ax6,
                                                      0, 1.3, 0.1, -15, 60, 5, "SOC", "Temp.", lgnd) 
                                        # Data output option
                                        if self.saveok.isChecked() and save_file_name:
                                            Chgtemp[1].Profile.to_excel(
                                                writer,
                                                startcol=writecolno,
                                                index=False,
                                                header=[
                                                    headername + "Time(min)",
                                                    headername + "SOC",
                                                    headername + "Energy",
                                                    headername + "Voltage",
                                                    headername + "Crate",
                                                    headername + "dQdV",
                                                    headername + "dVdQ",
                                                    headername + "Temp."
                                                ])
                                            writecolno = writecolno + 8
                                title = Chgnamelist[-2] + "=" + "%04d" % CycNo
                                plt.suptitle(title, fontsize= 15, fontweight='bold')
                                if len(all_data_name) != 0:
                                    Chg_ax1.legend(loc="lower right")
                                    Chg_ax2.legend(loc="lower right")
                                    Chg_ax3.legend(loc="lower right")
                                    Chg_ax4.legend(loc="upper right")
                                    Chg_ax5.legend(loc="upper right")
                                    Chg_ax6.legend(loc="upper right")
                                else:
                                    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
                        tab_layout.addWidget(toolbar)
                        tab_layout.addWidget(canvas)
                        self.cycle_tab.addTab(tab, str(tab_no))
                        self.cycle_tab.setCurrentWidget(tab)
                        tab_no = tab_no + 1
                        plt.tight_layout(pad=1, w_pad=1, h_pad=1)
                        output_fig(self.figsaveok, title)
        if self.saveok.isChecked() and save_file_name:
            writer.close()
        plt.tight_layout(pad=1, w_pad=1, h_pad=1)
        self.progressBar.setValue(100)
        plt.close()

    def dchg_confirm_button(self):
        self.DchgConfirm.setDisabled(True)
        firstCrate, mincapacity, CycleNo, smoothdegree, mincrate, dqscale, dvscale = self.Profile_ini_set()
        # 용량 선정 관련
        global writer
        foldercount, chnlcount, cyccount, writecolno = 0, 0, 0, 0
        root = Tk()
        root.withdraw()
        pne_path = self.pne_path_setting()
        all_data_folder = pne_path[0]
        all_data_name = pne_path[1]
        self.DchgConfirm.setEnabled(True)
        if self.saveok.isChecked():
            save_file_name = filedialog.asksaveasfilename(initialdir="D://", title="Save File Name", defaultextension=".xlsx")
            if save_file_name:
                writer = pd.ExcelWriter(save_file_name, engine="xlsxwriter")
        if self.ect_saveok.isChecked():
            # save_file_name = filedialog.asksaveasfilename(initialdir="D://", title="Save File Name", defaultextension=".csv")
            save_file_name = filedialog.asksaveasfilename(initialdir="D://", title="Save File Name")
        tab_no = 0
        for i, cyclefolder in enumerate(all_data_folder):
            if os.path.isdir(cyclefolder):
                subfolder = [f.path for f in os.scandir(cyclefolder) if f.is_dir()]
                foldercountmax = len(all_data_folder)
                foldercount = foldercount + 1
                if self.CycProfile.isChecked():
                    for FolderBase in subfolder:
                        chnlcount = chnlcount + 1
                        chnlcountmax = len(subfolder)
                        if "Pattern" not in FolderBase:
                            fig, ((Chg_ax1, Chg_ax2, Chg_ax3) ,(Chg_ax4, Chg_ax5, Chg_ax6)) = plt.subplots(
                                nrows=2, ncols=3, figsize=(14, 10))
                            tab = QtWidgets.QWidget()
                            tab_layout = QtWidgets.QVBoxLayout(tab)
                            canvas = FigureCanvas(fig)
                            toolbar = NavigationToolbar(canvas, None)
                            for CycNo in CycleNo:
                                cyccountmax = len(CycleNo)
                                cyccount = cyccount + 1
                                progressdata = progress(foldercount, foldercountmax, chnlcount, chnlcountmax, cyccount, cyccountmax)
                                self.progressBar.setValue(int(progressdata))
                                Dchgnamelist = FolderBase.split("\\")
                                headername = Dchgnamelist[-2] + ", " + Dchgnamelist[-1] + ", " + str(CycNo) + "cy, "
                                lgnd = "%04d" % CycNo
                                if not check_cycler(cyclefolder):
                                    Dchgtemp = toyo_dchg_Profile_data(FolderBase, CycNo, mincapacity, mincrate, firstCrate,
                                                                      smoothdegree)
                                else:
                                    Dchgtemp = pne_dchg_Profile_data(FolderBase, CycNo, mincapacity, mincrate, firstCrate,
                                                                     smoothdegree)
                                if len(all_data_name) == 0:
                                    temp_lgnd = ""
                                else:	
                                    temp_lgnd = all_data_name[i] + " " + lgnd
                                if hasattr(Dchgtemp[1], "Profile"):
                                    if len(Dchgtemp[1].Profile) > 2:
                                        self.capacitytext.setText(str(Dchgtemp[0]))
                                        graph_profile(Dchgtemp[1].Profile.SOC, Dchgtemp[1].Profile.Vol, Chg_ax1,
                                                      0, 1.3, 0.1, self.vol_y_hlimit, self.vol_y_llimit, self.vol_y_gap, "DOD", "Voltage(V)", temp_lgnd)
                                        graph_profile(Dchgtemp[1].Profile.SOC, Dchgtemp[1].Profile.Vol, Chg_ax3,
                                                      0, 1.3, 0.1, self.vol_y_hlimit, self.vol_y_llimit, self.vol_y_gap, "DOD", "Voltage(V)", temp_lgnd)
                                        graph_profile(Dchgtemp[1].Profile.dQdV, Dchgtemp[1].Profile.Vol, Chg_ax2,
                                                      -5 * dqscale, 0.5 * dqscale, 0.5 * dqscale, self.vol_y_hlimit, self.vol_y_llimit, self.vol_y_gap,
                                                      "dQdV", "Voltage(V)", temp_lgnd)
                                        graph_profile(Dchgtemp[1].Profile.SOC, Dchgtemp[1].Profile.Crate, Chg_ax5,
                                                      0, 1.3, 0.1, 0, 3.4, 0.2, "SOC", "C-rate", temp_lgnd)
                                        graph_profile(Dchgtemp[1].Profile.SOC, Dchgtemp[1].Profile.dVdQ, Chg_ax4,
                                                      0, 1.3, 0.1, -5 * dvscale, 0.5 * self.dvscale, 0.5 * self.dvscale,
                                                      "DOD", "dVdQ", temp_lgnd)
                                        graph_profile(Dchgtemp[1].Profile.SOC, Dchgtemp[1].Profile.Temp, Chg_ax6,
                                                      0, 1.3, 0.1, -15, 60, 5, "DOD", "Temp.", lgnd) # Data output option
                                        if self.saveok.isChecked() and save_file_name:
                                            Dchgtemp[1].Profile.to_excel(
                                                writer,
                                                startcol=writecolno,
                                                index=False,
                                                header=[
                                                    headername + "Time(min)",
                                                    headername + "DOD",
                                                    headername + "Energy",
                                                    headername + "Voltage",
                                                    headername + "Crate",
                                                    headername + "dQdV",
                                                    headername + "dVdQ",
                                                    headername + "Temp."
                                                ])
                                            writecolno = writecolno + 8
                                        if self.ect_saveok.isChecked() and save_file_name:
                                            Dchgtemp[1].Profile["TimeSec"] = Dchgtemp[1].Profile.TimeMin * 60
                                            Dchgtemp[1].Profile["Curr"] = Dchgtemp[1].Profile.Crate * Dchgtemp[0] / 1000
                                            continue_df = Dchgtemp[1].Profile.loc[:,["TimeSec", "Vol", "Curr", "Temp"]]
                                            # 각 열을 소수점 자리수에 맞게 반올림
                                            continue_df['TimeSec'] = continue_df['TimeSec'].round(1)  # 소수점 1자리
                                            continue_df['Vol'] = continue_df['Vol'].round(4)           # 소수점 4자리
                                            continue_df['Curr'] = continue_df['Curr'].round(4)         # 소수점 4자리
                                            continue_df['Temp'] = continue_df['Temp'].round(1)         # 소수점 1자리
                                            continue_df.to_csv(save_file_name +"_"+ "%04d" % CycNo + ".csv", index=False, sep=',',
                                                                header=["time(s)",
                                                                        "Voltage(V)",
                                                                        "Current(A)",
                                                                        "Temp."])
                                title = Dchgnamelist[-2] + "=" + Dchgnamelist[-1]
                                plt.suptitle(title, fontsize= 15, fontweight='bold')
                                if len(all_data_name) != 0:
                                    Chg_ax1.legend(loc="lower left")
                                    Chg_ax2.legend(loc="upper left")
                                    Chg_ax3.legend(loc="lower left")
                                    Chg_ax4.legend(loc="lower left")
                                    Chg_ax5.legend(loc="upper right")
                                    Chg_ax6.legend(loc="upper right")
                                else:
                                    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
                                plt.tight_layout(pad=1, w_pad=1, h_pad=1)
                            tab_layout.addWidget(toolbar)
                            tab_layout.addWidget(canvas)
                            self.cycle_tab.addTab(tab, str(tab_no))
                            self.cycle_tab.setCurrentWidget(tab)
                            tab_no = tab_no + 1
                            plt.tight_layout(pad=1, w_pad=1, h_pad=1)
                            output_fig(self.figsaveok, title)
                else:
                    for CycNo in CycleNo:
                        chnlcount = chnlcount + 1
                        chnlcountmax = len(subfolder)
                        fig, ((Chg_ax1, Chg_ax2, Chg_ax3) ,(Chg_ax4, Chg_ax5, Chg_ax6)) = plt.subplots(
                            nrows=2, ncols=3, figsize=(14, 10))
                        tab = QtWidgets.QWidget()
                        tab_layout = QtWidgets.QVBoxLayout(tab)
                        canvas = FigureCanvas(fig)
                        toolbar = NavigationToolbar(canvas, None)
                        for FolderBase in subfolder:
                            if "Pattern" not in FolderBase:
                                cyccountmax = len(CycleNo)
                                cyccount = cyccount + 1
                                progressdata = progress(foldercount, foldercountmax, cyccount, cyccountmax, chnlcount, chnlcountmax)
                                self.progressBar.setValue(int(progressdata))
                                Dchgnamelist = FolderBase.split("\\")
                                headername = Dchgnamelist[-2] + ", " + Dchgnamelist[-1] + ", " + str(CycNo) + "cy, "
                                lgnd = Dchgnamelist[-1]
                                if not check_cycler(cyclefolder):
                                    Dchgtemp = toyo_dchg_Profile_data(FolderBase, CycNo, mincapacity, mincrate, firstCrate, smoothdegree)
                                else:
                                    Dchgtemp = pne_dchg_Profile_data(FolderBase, CycNo, mincapacity, mincrate, firstCrate, smoothdegree)
                                if len(all_data_name) == 0:
                                    temp_lgnd = ""
                                else:	
                                    temp_lgnd = all_data_name[i] + " " + lgnd
                                if hasattr(Dchgtemp[1], "Profile"):
                                    if len(Dchgtemp[1].Profile) > 2:
                                        self.capacitytext.setText(str(Dchgtemp[0]))
                                        graph_profile(Dchgtemp[1].Profile.SOC, Dchgtemp[1].Profile.Vol, Chg_ax1,
                                                      0, 1.3, 0.1, self.vol_y_hlimit, self.vol_y_llimit, self.vol_y_gap, "DOD", "Voltage(V)", temp_lgnd)
                                        graph_profile(Dchgtemp[1].Profile.SOC, Dchgtemp[1].Profile.Vol, Chg_ax3,
                                                      0, 1.3, 0.1, self.vol_y_hlimit, self.vol_y_llimit, self.vol_y_gap, "DOD", "Voltage(V)", temp_lgnd)
                                        graph_profile(Dchgtemp[1].Profile.dQdV, Dchgtemp[1].Profile.Vol, Chg_ax2,
                                                      -5 * dqscale, 0.5 * dqscale, 0.5 * dqscale, self.vol_y_hlimit, self.vol_y_llimit, self.vol_y_gap,
                                                      "dQdV", "Voltage(V)", temp_lgnd)
                                        graph_profile(Dchgtemp[1].Profile.SOC, Dchgtemp[1].Profile.Crate, Chg_ax5,
                                                      0, 1.3, 0.1, 0, 3.4, 0.2, "SOC", "C-rate", temp_lgnd)
                                        graph_profile(Dchgtemp[1].Profile.SOC, Dchgtemp[1].Profile.dVdQ, Chg_ax4,
                                                      0, 1.3, 0.1, -5 * dvscale, 0.5 * self.dvscale, 0.5 * self.dvscale,
                                                      "DOD", "dVdQ", temp_lgnd)
                                        graph_profile(Dchgtemp[1].Profile.SOC, Dchgtemp[1].Profile.Temp, Chg_ax6,
                                                      0, 1.3, 0.1, -15, 60, 5, "DOD", "Temp.", lgnd) 
                                        # Data output option
                                        if self.saveok.isChecked() and save_file_name:
                                            Dchgtemp[1].Profile.to_excel(
                                                writer,
                                                startcol=writecolno,
                                                index=False,
                                                header=[
                                                    headername + "Time(min)",
                                                    headername + "DOD",
                                                    headername + "Energy",
                                                    headername + "Voltage",
                                                    headername + "Crate",
                                                    headername + "dQdV",
                                                    headername + "dVdQ",
                                                    headername + "Temp."
                                                ])
                                            writecolno = writecolno + 8
                                        if self.ect_saveok.isChecked() and save_file_name:
                                            Dchgtemp[1].Profile["TimeSec"] = Dchgtemp[1].Profile.TimeMin * 60
                                            Dchgtemp[1].Profile["Curr"] = Dchgtemp[1].Profile.Crate * Dchgtemp[0] / 1000
                                            continue_df = Dchgtemp[1].Profile.loc[:,["TimeSec", "Vol", "Curr", "Temp"]]
                                            # 각 열을 소수점 자리수에 맞게 반올림
                                            continue_df['TimeSec'] = continue_df['TimeSec'].round(1)  # 소수점 1자리
                                            continue_df['Vol'] = continue_df['Vol'].round(4)           # 소수점 4자리
                                            continue_df['Curr'] = continue_df['Curr'].round(4)         # 소수점 4자리
                                            continue_df['Temp'] = continue_df['Temp'].round(1)         # 소수점 1자리
                                            continue_df.to_csv(save_file_name + "_" + Dchgnamelist[-1] + ".csv", index=False, sep=',',
                                                                header=["time(s)",
                                                                        "Voltage(V)",
                                                                        "Current(A)",
                                                                        "Temp."])
                                title = Dchgnamelist[-2] + "=" + "%04d" % CycNo
                                plt.suptitle(title, fontsize= 15, fontweight='bold')
                                if len(all_data_name) != 0:
                                    Chg_ax1.legend(loc="lower left")
                                    Chg_ax2.legend(loc="upper left")
                                    Chg_ax3.legend(loc="lower left")
                                    Chg_ax4.legend(loc="lower left")
                                    Chg_ax5.legend(loc="upper right")
                                    Chg_ax6.legend(loc="upper right")
                                else:
                                    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
                                plt.tight_layout(pad=1, w_pad=1, h_pad=1)
                        tab_layout.addWidget(toolbar)
                        tab_layout.addWidget(canvas)
                        self.cycle_tab.addTab(tab, str(tab_no))
                        self.cycle_tab.setCurrentWidget(tab)
                        tab_no = tab_no + 1
                        plt.tight_layout(pad=1, w_pad=1, h_pad=1)
                        output_fig(self.figsaveok, title)
                plt.tight_layout(pad=1, w_pad=1, h_pad=1)
                plt.close()
            else:
                err_msg('파일 or 폴더 없음!!','파일 or 폴더 없음!!')
                return
        self.progressBar.setValue(100)
        if self.saveok.isChecked() and save_file_name:
            writer.close()

    def continue_confirm_button(self):
        if self.chk_ectpath.isChecked():
            self.ect_confirm_button()
        else:
            self.pro_continue_confirm_button()

    def ect_confirm_button(self):
        self.ContinueConfirm.setDisabled(True)
        firstCrate, mincapacity, CycleNo, smoothdegree, mincrate, dqscale, dvscale = self.Profile_ini_set()
        all_data_name = []
        # 용량 선정 관련
        global writer
        write_column_num, write_column_num2, folder_count, chnlcount, cyccount = 0, 0, 0, 0, 0
        root = Tk()
        root.withdraw()
        datafilepath = filedialog.askopenfilename(initialdir="d://", title="Choose Test files")
        if datafilepath:
            cycle_path = pd.read_csv(datafilepath, sep="\t", engine="c", encoding="UTF-8", skiprows=1, on_bad_lines='skip')
            ect_path = np.array(cycle_path.path.tolist())
            ect_cycle = np.array(cycle_path.cycle.tolist())
            ect_CD = np.array(cycle_path.CD.tolist())
            ect_save = np.array(cycle_path.save.tolist())
            if (self.inicaprate.isChecked()) and ("mAh" in datafilepath):
                self.mincapacity = name_capacity(datafilepath)
                self.capacitytext.setText(str(self.mincapacity))
        self.ContinueConfirm.setEnabled(True)
        tab_no = 0
        for i, cyclefolder in enumerate(ect_path):
            chg_dchg_dcir_no = list((ect_cycle[i].split(" ")))
            if os.path.isdir(cyclefolder):
                subfolder = [f.path for f in os.scandir(cyclefolder) if f.is_dir()]
                foldercountmax = len(ect_path)
                folder_count = folder_count + 1
                for FolderBase in subfolder:
                    for dcir_continue_step in chg_dchg_dcir_no:
                        if "-" in dcir_continue_step:
                            Step_CycNo, Step_CycEnd = map(int, dcir_continue_step.split("-"))
                        else:
                            Step_CycNo, Step_CycEnd = int(dcir_continue_step), int(dcir_continue_step)
                        CycleNo = range(Step_CycNo, Step_CycEnd + 1)
                        if "Pattern" not in FolderBase:
                            fig, ((step_ax1, step_ax2, step_ax3) ,(step_ax4, step_ax5, step_ax6)) = plt.subplots( nrows=2, ncols=3, figsize=(14, 10))
                            tab = QtWidgets.QWidget()
                            tab_layout = QtWidgets.QVBoxLayout(tab)
                            canvas = FigureCanvas(fig)
                            toolbar = NavigationToolbar(canvas, None)
                            progressdata = progress(folder_count, foldercountmax, 1, 1, 1, 1)
                            self.progressBar.setValue(int(progressdata))
                            step_namelist = FolderBase.split("\\")
                            headername = step_namelist[-2] + ", " + step_namelist[-1] + ", " + str( Step_CycNo)
                            headername = headername + "-" + str(Step_CycEnd) + "cy, "
                            if self.CycProfile.isChecked():
                                lgnd = "%04d" % Step_CycNo
                            else:
                                lgnd = step_namelist[-1]
                            temp = pne_Profile_continue_data(FolderBase, Step_CycNo, Step_CycEnd, mincapacity, firstCrate, ect_CD[i])
                            if len(all_data_name) == 0:
                                temp_lgnd = ""
                            else:	
                                temp_lgnd = all_data_name[i]
                            if hasattr(temp[1], "stepchg"):
                                if len(temp[1].stepchg) > 2:
                                    self.capacitytext.setText(str(temp[0]))
                                    graph_continue(temp[1].stepchg.TimeMin, temp[1].stepchg.Vol, step_ax1, 2.0, 4.8, 0.2, "Time(min)", "Voltage(V)",temp_lgnd)
                                    graph_continue(temp[1].stepchg.TimeMin, temp[1].stepchg.Vol, step_ax4, 2.0, 4.8, 0.2, "Time(min)", "Voltage(V)",temp_lgnd)
                                    graph_continue(temp[1].stepchg.TimeMin, temp[1].stepchg.Crate, step_ax2, 0, 3.2, 0.2, "Time(min)", "C-rate",temp_lgnd)
                                    graph_continue(temp[1].stepchg.TimeMin, temp[1].stepchg.Crate, step_ax5, -3.0, 0.2, 0.2, "Time(min)", "C-rate",temp_lgnd)
                                    graph_continue(temp[1].stepchg.TimeMin, temp[1].stepchg.SOC, step_ax3, 0, 1.2, 0.1, "Time(min)", "SOC", temp_lgnd)
                                    graph_continue(temp[1].stepchg.TimeMin, temp[1].stepchg.Temp, step_ax6, -15, 60, 5, "Time(min)", "Temp.", lgnd)
                                    # Data output option
                                    continue_df = temp[1].stepchg.loc[:,["TimeSec", "Vol", "Curr", "Temp"]]
                                    # 각 열을 소수점 자리수에 맞게 반올림
                                    continue_df['TimeSec'] = continue_df['TimeSec'].round(1)  # 소수점 1자리
                                    continue_df['Vol'] = continue_df['Vol'].round(4)           # 소수점 4자리
                                    continue_df['Curr'] = continue_df['Curr'].round(4)         # 소수점 4자리
                                    continue_df['Temp'] = continue_df['Temp'].round(1)         # 소수점 1자리
                                    continue_df.to_csv(("D:\\" + ect_save[i] + ".csv"), index=False, sep=',',
                                                        header=["time(s)",
                                                                "Voltage(V)",
                                                                "Current(A)",
                                                                "Temp."])
                            title = step_namelist[-2] + "=" + "%04d" % Step_CycNo
                            plt.suptitle(title, fontsize= 15, fontweight='bold')
                            if len(all_data_name) != 0:
                                step_ax1.legend(loc="lower left")
                                step_ax2.legend(loc="lower right")
                                step_ax3.legend(loc="upper right")
                                step_ax4.legend(loc="lower right")
                                step_ax5.legend(loc="lower left")
                                step_ax6.legend(loc="upper right")
                            else:
                                plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
                            tab_layout.addWidget(toolbar)
                            tab_layout.addWidget(canvas)
                            self.cycle_tab.addTab(tab, str(tab_no))
                            self.cycle_tab.setCurrentWidget(tab)
                            tab_no = tab_no + 1
                            plt.tight_layout(pad=1, w_pad=1, h_pad=1)
                            output_fig(self.figsaveok, ect_save[i])
        self.progressBar.setValue(100)
        plt.tight_layout(pad=1, w_pad=1, h_pad=1)
        plt.close()

    def pro_continue_confirm_button(self):
        self.ContinueConfirm.setDisabled(True)
        firstCrate, mincapacity, CycleNo, smoothdegree, mincrate, dqscale, dvscale = self.Profile_ini_set()
        all_data_name = []
        # 용량 선정 관련
        global writer
        if "-" in self.stepnum.toPlainText():
            write_column_num, write_column_num2, folder_count, chnlcount, cyccount = 0, 0, 0, 0, 0
            root = Tk()
            root.withdraw()
            pne_path = self.pne_path_setting()
            all_data_folder = pne_path[0]
            all_data_name = pne_path[1]
            if self.saveok.isChecked():
                save_file_name = filedialog.asksaveasfilename(initialdir="D://", title="Save File Name", defaultextension=".xlsx")
                if save_file_name:
                    writer = pd.ExcelWriter(save_file_name, engine="xlsxwriter")
            if self.ect_saveok.isChecked():
                # save_file_name = filedialog.asksaveasfilename(initialdir="D://", title="Save File Name", defaultextension=".csv")
                save_file_name = filedialog.asksaveasfilename(initialdir="D://", title="Save File Name")
            self.ContinueConfirm.setEnabled(True)
            chg_dchg_dcir_no = list((self.stepnum.toPlainText().split(" ")))
            tab_no = 0
            for i, cyclefolder in enumerate(all_data_folder):
                if os.path.isdir(cyclefolder):
                    subfolder = [f.path for f in os.scandir(cyclefolder) if f.is_dir()]
                    foldercountmax = len(all_data_folder)
                    folder_count = folder_count + 1
                    for FolderBase in subfolder:
                        chnlcount = chnlcount + 1
                        chnlcountmax = len(subfolder)
                        for dcir_continue_step in chg_dchg_dcir_no:
                            if "-" in dcir_continue_step:
                                Step_CycNo, Step_CycEnd = map(int, dcir_continue_step.split("-"))
                                CycleNo = range(Step_CycNo, Step_CycEnd + 1)
                                if "Pattern" not in FolderBase:
                                    fig, ((step_ax1, step_ax2, step_ax3) ,(step_ax4, step_ax5, step_ax6)) = plt.subplots( nrows=2, ncols=3, figsize=(14, 10))
                                    tab = QtWidgets.QWidget()
                                    tab_layout = QtWidgets.QVBoxLayout(tab)
                                    canvas = FigureCanvas(fig)
                                    toolbar = NavigationToolbar(canvas, None)
                                    cyccountmax = len(CycleNo)
                                    cyccount = cyccount + 1
                                    progressdata = progress(folder_count, foldercountmax, cyccount, cyccountmax, chnlcount, chnlcountmax)
                                    self.progressBar.setValue(int(progressdata))
                                    step_namelist = FolderBase.split("\\")
                                    headername = step_namelist[-2] + ", " + step_namelist[-1] + ", " + str( Step_CycNo)
                                    headername = headername + "-" + str(Step_CycEnd) + "cy, "
                                    if self.CycProfile.isChecked():
                                        lgnd = "%04d" % Step_CycNo
                                    else:
                                        lgnd = step_namelist[-1]
                                    if not check_cycler(cyclefolder):
                                        err_msg("Toyo는 준비 중", "토요는 시간나면 추가할께요 ^^;")
                                    else:
                                        temp = pne_Profile_continue_data(FolderBase, Step_CycNo, Step_CycEnd, mincapacity, firstCrate, "")
                                        if len(all_data_name) == 0:
                                            temp_lgnd = ""
                                        else:	
                                            temp_lgnd = all_data_name[i]
                                        if hasattr(temp[1], "stepchg"):
                                            if len(temp[1].stepchg) > 2:
                                                self.capacitytext.setText(str(temp[0]))
                                                graph_continue(temp[1].stepchg.TimeMin, temp[1].stepchg.Vol, step_ax1,
                                                               2.0, 4.8, 0.2, "Time(min)", "Voltage(V)",temp_lgnd)
                                                graph_continue(temp[1].stepchg.TimeMin, temp[1].stepchg.Vol, step_ax4,
                                                               2.0, 4.8, 0.2, "Time(min)", "Voltage(V)",temp_lgnd)
                                                graph_continue(temp[1].stepchg.TimeMin, temp[1].stepchg.OCV, step_ax4,
                                                               2.0, 4.8, 0.2, "Time(min)", "OCV/CCV", "OCV_" + temp_lgnd, "o")
                                                graph_continue(temp[1].stepchg.TimeMin, temp[1].stepchg.CCV, step_ax4,
                                                               2.0, 4.8, 0.2, "Time(min)", "OCV/CCV", "CCV_" + temp_lgnd, "o")
                                                graph_continue(temp[1].stepchg.TimeMin, temp[1].stepchg.Crate, step_ax2,
                                                               -1.8, 1.7, 0.2, "Time(min)", "C-rate",temp_lgnd)
                                                graph_continue(temp[2].AccCap, temp[2].OCV, step_ax5, 2.0, 4.8, 0.2,
                                                               "SOC", "OCV/CCV", "OCV_" + temp_lgnd, "o") 
                                                graph_continue(temp[2].AccCap, temp[2].CCV, step_ax5, 2.0, 4.8, 0.2,
                                                               "SOC", "OCV/CCV", "CCV_" + temp_lgnd, "o")
                                                graph_continue(temp[1].stepchg.TimeMin, temp[1].stepchg.SOC, step_ax3,
                                                               0, 1.2, 0.1, "Time(min)", "SOC", temp_lgnd)
                                                graph_continue(temp[1].stepchg.TimeMin, temp[1].stepchg.Temp, step_ax6,
                                                               -15, 60, 5, "Time(min)", "Temp.", lgnd)
                                                # Data output option
                                                if self.saveok.isChecked() and save_file_name:
                                                    temp[1].stepchg = temp[1].stepchg.loc[:,["TimeSec", "Vol", "Curr","OCV", "CCV",
                                                                                             "Crate", "SOC", "Temp"]]
                                                    temp[1].stepchg.to_excel(writer, sheet_name="Profile", startcol=write_column_num,
                                                                             index=False,
                                                                             header=[headername + "time(s)",
                                                                                     headername + "Voltage(V)",
                                                                                     headername + "Current(A)",
                                                                                     headername + "OCV",
                                                                                     headername + "CCV",
                                                                                     headername + "Crate",
                                                                                     headername + "SOC",
                                                                                     headername + "Temp."])
                                                    write_column_num = write_column_num + 8
                                                    temp[2].to_excel(writer, sheet_name="OCV_CCV", startcol=write_column_num2,
                                                                     index=False,
                                                                     header=[headername + "SOC",
                                                                             headername + "OCV",
                                                                             headername + "CCV"])
                                                    write_column_num2 = write_column_num2 + 3
                                                if self.ect_saveok.isChecked() and save_file_name:
                                                    temp[1].stepchg["TimeSec"] = temp[1].stepchg.TimeMin * 60
                                                    temp[1].stepchg["Curr"] = temp[1].stepchg.Crate * temp[0] / 1000
                                                    continue_df = temp[1].stepchg.loc[:,["TimeSec", "Vol", "Curr", "Temp"]]
                                                    # 각 열을 소수점 자리수에 맞게 반올림
                                                    continue_df['TimeSec'] = continue_df['TimeSec'].round(1)  # 소수점 1자리
                                                    continue_df['Vol'] = continue_df['Vol'].round(4)           # 소수점 4자리
                                                    continue_df['Curr'] = continue_df['Curr'].round(4)         # 소수점 4자리
                                                    continue_df['Temp'] = continue_df['Temp'].round(1)         # 소수점 1자리
                                                    continue_df.to_csv((save_file_name + "_" + "%04d" % tab_no + ".csv"), index=False, sep=',',
                                                                        header=["time(s)",
                                                                                "Voltage(V)",
                                                                                "Current(A)",
                                                                                "Temp."])
                                        if self.CycProfile.isChecked():
                                            title = step_namelist[-2] + "=" + step_namelist[-1]
                                        else:
                                            title = step_namelist[-2] + "=" + "%04d" % Step_CycNo
                                        plt.suptitle(title, fontsize= 15, fontweight='bold')
                                        if len(all_data_name) != 0:
                                            step_ax1.legend(loc="lower left")
                                            step_ax2.legend(loc="lower right")
                                            step_ax3.legend(loc="upper right")
                                            step_ax4.legend(loc="lower right")
                                            step_ax5.legend(loc="lower left")
                                            step_ax6.legend(loc="upper right")
                                        else:
                                            plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
                                        if self.CycProfile.isChecked():
                                            tab_layout.addWidget(toolbar)
                                            tab_layout.addWidget(canvas)
                                            self.cycle_tab.addTab(tab, str(tab_no))
                                            self.cycle_tab.setCurrentWidget(tab)
                                            # tab_no = tab_no + 1
                                            plt.tight_layout(pad=1, w_pad=1, h_pad=1)
                                            output_fig(self.figsaveok, title)
                                        tab_layout.addWidget(toolbar)
                                        tab_layout.addWidget(canvas)
                                        self.cycle_tab.addTab(tab, str(tab_no))
                                        self.cycle_tab.setCurrentWidget(tab)
                                        tab_no = tab_no + 1
                                        plt.tight_layout(pad=1, w_pad=1, h_pad=1)
                                        output_fig(self.figsaveok, title)
            if self.saveok.isChecked() and save_file_name:
                writer.close()
            self.progressBar.setValue(100)
            plt.tight_layout(pad=1, w_pad=1, h_pad=1)
            plt.close()
        else:
            err_msg('Step 에러','Step에 3-5 같은 연속 형식으로 넣어주세요!')
            self.ContinueConfirm.setEnabled(True)

    def dcir_confirm_button(self):
        firstCrate, mincapacity, CycleNo, smoothdegree, mincrate, dqscale, dvscale = self.Profile_ini_set()
        # 용량 선정 관련
        root = Tk()
        root.withdraw()
        global writer
        # if "-" in self.stepnum.toPlainText():
        write_column_num, folder_count, chnlcount, cyccount = 0, 0, 0, 0
        self.DCIRConfirm.setDisabled(True)
        pne_path = self.pne_path_setting()
        all_data_folder = pne_path[0]
        all_data_name = pne_path[1]
        if self.saveok.isChecked():
            save_file_name = filedialog.asksaveasfilename(initialdir="D://", title="Save File Name", defaultextension=".xlsx")
            if save_file_name:
                writer = pd.ExcelWriter(save_file_name, engine="xlsxwriter")
        self.DCIRConfirm.setEnabled(True)
        # chg_dchg_dcir_no = list((self.stepnum.toPlainText().split(" ")))
        chg_tab_no, dchg_tab_no = 0, 0
        for i, cyclefolder in enumerate(all_data_folder):
            if os.path.isdir(cyclefolder):
                subfolder = [f.path for f in os.scandir(cyclefolder) if f.is_dir()]
                foldercountmax = len(all_data_folder)
                folder_count = folder_count + 1
                for FolderBase in subfolder:
                    chg_dchg_dcir_no = pne_dcir_chk_cycle(FolderBase)
                    chnlcount = chnlcount + 1
                    chnlcountmax = len(subfolder)
                    if chg_dchg_dcir_no is not None:
                        for dcir_continue_step in chg_dchg_dcir_no:
                            if "-" in dcir_continue_step:
                                Step_CycNo, Step_CycEnd = map(int, dcir_continue_step.split("-"))
                                CycleNo = range(Step_CycNo, Step_CycEnd + 1)
                                if "Pattern" not in FolderBase:
                                    fig, ((step_ax1, step_ax3), (step_ax2, step_ax4)) = plt.subplots(
                                        nrows=2, ncols=2, figsize=(14, 8))
                                    tab = QtWidgets.QWidget()
                                    tab_layout = QtWidgets.QVBoxLayout(tab)
                                    canvas = FigureCanvas(fig)
                                    toolbar = NavigationToolbar(canvas, None)
                                    cyccountmax = len(CycleNo)
                                    cyccount = cyccount + 1
                                    progressdata = progress(folder_count, foldercountmax, chnlcount, chnlcountmax, cyccount, cyccountmax)
                                    self.progressBar.setValue(int(progressdata))
                                    step_namelist = FolderBase.split("\\")
                                    headername = step_namelist[-2] + ", " + step_namelist[-1]
                                    if self.CycProfile.isChecked():
                                        lgnd = "%04d" % Step_CycNo
                                    else:
                                        lgnd = step_namelist[-1]
                                    if not check_cycler(cyclefolder):
                                        err_msg("PNE 충방전기 사용 요청", "DCIR은 PNE 충방전기를 사용하여 측정 부탁 드립니다.")
                                    else:
                                        temp = pne_dcir_Profile_data(FolderBase, Step_CycNo, Step_CycEnd, mincapacity, firstCrate)
                                        if (temp is not None) and hasattr(temp[1], "AccCap"):
                                            if len(temp[1]) > 2:
                                                self.capacitytext.setText(str(temp[0]))
                                                graph_soc_continue(temp[1].SOC, temp[1].OCV, step_ax1, 2.0, 4.8, 0.2, "SOC", "OCV/CCV", "OCV", "o")
                                                graph_soc_continue(temp[1].SOC, temp[1].rOCV, step_ax1, 2.0, 4.8, 0.2, "SOC", "OCV/CCV", "rOCV", "o")
                                                graph_soc_continue(temp[1].SOC, temp[1].CCV, step_ax1, 2.0, 4.8, 0.2, "SOC", "OCV/CCV","CCV", "o")
                                                graph_soc_dcir(temp[1].SOC, temp[1].iloc[:, 7], step_ax2, "SOC", "DCIR(mΩ)", " 0.1s DCIR", "o")
                                                graph_soc_dcir(temp[1].SOC, temp[1].iloc[:, 8], step_ax2, "SOC", "DCIR(mΩ)", " 1.0s DCIR", "o")
                                                graph_soc_dcir(temp[1].SOC, temp[1].iloc[:, 9], step_ax2, "SOC", "DCIR(mΩ)", "10.0s DCIR", "o")
                                                graph_soc_dcir(temp[1].SOC, temp[1].iloc[:, 10], step_ax2, "SOC", "DCIR(mΩ)", "20.0s DCIR", "o")
                                                graph_soc_dcir(temp[1].SOC, temp[1].RSS, step_ax2, "SOC", "DCIR(mΩ)", "RSS DCIR", "o")
                                                graph_continue(temp[1].OCV, temp[1].SOC, step_ax3, -20, 120, 10, "Voltage (V)", "SOC","OCV", "o")
                                                graph_continue(temp[1].CCV, temp[1].SOC, step_ax3, -20, 120, 10, "Voltage (V)", "SOC","CCV", "o")
                                                graph_dcir(temp[1].OCV, temp[1].iloc[:, 7], step_ax4, "OCV", "DCIR(mΩ)", " 0.1s DCIR", "o")
                                                graph_dcir(temp[1].OCV, temp[1].iloc[:, 8], step_ax4, "OCV", "DCIR(mΩ)", " 1.0s DCIR", "o")
                                                graph_dcir(temp[1].OCV, temp[1].iloc[:, 9], step_ax4, "OCV", "DCIR(mΩ)", "10.0s DCIR", "o")
                                                graph_dcir(temp[1].OCV, temp[1].iloc[:, 10], step_ax4, "OCV", "DCIR(mΩ)", "20.0s DCIR", "o")
                                                graph_dcir(temp[1].OCV, temp[1].RSS, step_ax4, "OCV", "DCIR(mΩ)", "RSS DCIR", "o")
                                                # Data output option
                                                if self.saveok.isChecked() and save_file_name:
                                                    # temp[1] = temp[1].iloc[:,[1, 2, 4, 6, 7, 8, 9, 10, 5, 3]]
                                                    temp[1] = temp[1].iloc[:,[1, 2, 4, 7, 8, 9, 10, 5, 3]]
                                                    temp[1].to_excel(writer, sheet_name="DCIR", startcol=write_column_num, index=False,
                                                                        header=[headername + " Capacity(mAh)",
                                                                                headername + " SOC",
                                                                                headername + " OCV",
                                                                                # headername + " OCV_est",
                                                                                headername + "  0.1s DCIR",
                                                                                headername + "  1.0s DCIR",
                                                                                headername + " 10.0s DCIR",
                                                                                headername + " 20.0s DCIR",
                                                                                headername + " RSS",
                                                                                headername + " CCV"])
                                                    temp[2] = temp[2].iloc[:,[1, 2, 4, 7, 8, 9, 10, 5, 3]]
                                                    temp[2].to_excel(writer, sheet_name="RSQ", startcol=write_column_num, index=False,
                                                                        header=[headername + " Capacity(mAh)",
                                                                                headername + " SOC",
                                                                                headername + " OCV",
                                                                                # headername + " OCV_est",
                                                                                headername + "  0.1s DCIR RSQ",
                                                                                headername + "  1.0s DCIR RSQ",
                                                                                headername + " 10.0s DCIR RSQ",
                                                                                headername + " 20.0s DCIR RSQ",
                                                                                headername + " RSS",
                                                                                headername + " CCV"])
                                                    write_column_num = write_column_num + 9
                                                if self.CycProfile.isChecked():
                                                    title = step_namelist[-2] + "=" + step_namelist[-1]
                                                else:
                                                    title = step_namelist[-2] + "=" + "%04d" % Step_CycNo
                                                plt.suptitle(title, fontsize= 15, fontweight='bold')
                                                step_ax1.legend(loc="lower right")
                                                step_ax2.legend(loc="upper right")
                                                step_ax3.legend(loc="lower right")
                                                step_ax4.legend(loc="upper right")
                                                tab_layout.addWidget(toolbar)
                                                tab_layout.addWidget(canvas)
                                                if temp[1].iloc[0,2] == 100:
                                                    self.cycle_tab.addTab(tab, "dchg" + str(dchg_tab_no))
                                                    dchg_tab_no = dchg_tab_no + 1
                                                else:
                                                    self.cycle_tab.addTab(tab, "chg" + str(chg_tab_no))
                                                    chg_tab_no = chg_tab_no + 1
                                                self.cycle_tab.setCurrentWidget(tab)
                                                plt.tight_layout(pad=1, w_pad=1, h_pad=1)
                                                output_fig(self.figsaveok, title)
                                plt.tight_layout(pad=1, w_pad=1, h_pad=1)
                                plt.close()
        if self.saveok.isChecked() and save_file_name:
            writer.close()
        self.progressBar.setValue(100)
        # else:
        #     err_msg('Step 에러','Step에 3-5 같은 연속 형식으로 넣어주세요!')
        #     self.DCIRConfirm.setEnabled(True)

    def conn_disconn(self, conn_drive, drive_name):
        connect_change(conn_drive) if os.path.isdir(drive_name) else disconnect_change(conn_drive)

    def chk_network_drive(self):
        self.conn_disconn(self.mount_toyo, "z:")
        self.conn_disconn(self.mount_pne_1, "y:")
        self.conn_disconn(self.mount_pne_2, "x:")
        self.conn_disconn(self.mount_pne_3, "w:")
        self.conn_disconn(self.mount_pne_4, "v:")
        self.conn_disconn(self.mount_pne_5, "u:")

    def network_drive(self, driver, folder, id, pw):
        if not os.path.isdir(driver):
            if id == "":
                os.system('%SystemRoot%\\system32\\net use ' + driver + ' ' + folder +' ' + ' /persistent:no')
            else:
                os.system('%SystemRoot%\\system32\\net use ' + driver + ' ' + folder +' ' + pw + ' /user:' + id + ' /persistent:no')
        else:
            os.system('%SystemRoot%\\system32\\net use ' + driver + ' /delete /y')
        self.chk_network_drive()

    def mount_toyo_button(self):
        self.network_drive("z:",'"\\\\10.253.44.115\\TOYO-DATA Back Up Folder"', "sec", "qoxjfl1!")

    def mount_pne1_button(self):
        self.network_drive("y:",'"\\\\10.253.44.111\\PNE-Data"', "SAMSUNG", "qoxjfl1!")

    def mount_pne2_button(self):
        self.network_drive("x:",'"\\\\10.253.44.111\\PNE-Data2"', "SAMSUNG", "qoxjfl1!")

    def mount_pne3_button(self):
        self.network_drive("w:",'"\\\\10.252.130.113\\PNE-Data"', "", "")

    def mount_pne4_button(self):
        self.network_drive("v:",'"\\\\10.252.130.145\\PNE-Data"', "", "")

    def mount_pne5_button(self):
        self.network_drive("u:",'"\\\\10.252.130.162\\PNE-Data"', "", "")

    def mount_all_button(self):
        self.progressBar.setValue(0)
        if not os.path.isdir("z:"):
            self.network_drive("z:",'"\\\\10.253.44.115\\TOYO-DATA Back Up Folder"', "sec", "qoxjfl1!")
        self.progressBar.setValue(15)
        if not os.path.isdir("y:"):
            self.network_drive("y:",'"\\\\10.253.44.111\\PNE-Data"', "SAMSUNG", "qoxjfl1!")
        self.progressBar.setValue(30)
        if not os.path.isdir("x:"):
            self.network_drive("x:",'"\\\\10.253.44.111\\PNE-Data2"', "SAMSUNG", "qoxjfl1!")
        self.progressBar.setValue(45)
        if not os.path.isdir("w:"):
            self.network_drive("w:",'"\\\\10.252.130.113\\PNE-Data"', "", "")
        self.progressBar.setValue(60)
        if not os.path.isdir("v:"):
            self.network_drive("v:",'"\\\\10.252.130.145\\PNE-Data"', "", "")
        self.progressBar.setValue(75)
        if not os.path.isdir("u:"):
            self.network_drive("u:",'"\\\\10.252.130.162\\PNE-Data"', "", "")
        self.progressBar.setValue(90)
        self.chk_network_drive()
        self.progressBar.setValue(100)
        self.AllchnlData = pd.DataFrame()
        if self.saveok.isChecked():
            save_file_name = filedialog.asksaveasfilename(initialdir="D://", title="Save File Name", defaultextension=".xlsx")
            if save_file_name:
                self.progressBar.setValue(0)
                for i in range(0, 5):
                    self.toyo_data_make(i, self.toyo_cycler_name[i])
                    self.progressBar.setValue(int(((i + 1) / 5) * 20))
                for j in range(0, 26):
                    self.pne_data_make(j, self.pne_cycler_name[j])
                    self.progressBar.setValue(int(20 + ((j + 1) / 26) * 80))
                self.progressBar.setValue(100)
                writer = pd.ExcelWriter(save_file_name, engine="xlsxwriter")
                self.AllchnlData.to_excel(writer, index=False)
                writer.close()

    def unmount_all_button(self):
        self.progressBar.setValue(0)
        os.system(r'%SystemRoot%\system32\net use u: /delete /y')
        self.progressBar.setValue(15)
        os.system(r'%SystemRoot%\system32\net use v: /delete /y')
        self.progressBar.setValue(30)
        os.system(r'%SystemRoot%\system32\net use w: /delete /y')
        self.progressBar.setValue(45)
        os.system(r'%SystemRoot%\system32\net use x: /delete /y')
        self.progressBar.setValue(60)
        os.system(r'%SystemRoot%\system32\net use y: /delete /y')
        self.progressBar.setValue(75)
        os.system(r'%SystemRoot%\system32\net use z: /delete /y')
        self.progressBar.setValue(90)
        self.chk_network_drive()
        self.progressBar.setValue(100)

    def split_value0(self, x):
        if '_' in x:
            part = x.split('_')
        else:
            part = x.split(' ')
        return part[0]
    
    def split_value1(self, x):
        if '_' in x:
            part = x.split('_')
            if len(part) > 2:
                if part[2] == '00':
                    return "선행랩"
                else:
                    return part[2] + " 파트"
            else:
                return part[0]
        else:
            part = x.split(' ')
            if len(part) > 1:
                return part[1]
            else:
                return part[0]

    def split_value2(self, x):
        if '_' in x:
            part = x.split('_')
            if len(part) > 3:
                return part[3]
            else:
                return part[0]
        else:
            part = x.split(' ')
            if len(part) > 2:
                return part[2]
            else:
                return part[0]

    def toyo_base_data_make(self, toyo_num, blkname):
        # 경로 확인
        toyoworkpath = "z:\\Working\\"+self.toyo_blk_list[toyo_num]+"\\Chpatrn.cfg"
        if os.path.isfile(toyoworkpath):
            toyo_data = remove_end_comma(toyoworkpath)
            toyo_data = toyo_data[[7, 1, 5, 9]]
            toyo_data.columns = self.toyo_column_list[0:4]
            toyo_data.index = toyo_data.index + 1
            if toyo_num != 3:
                toyoworkpath2 = "z:\\Working\\"+self.toyo_blk_list[toyo_num]+"\\ExperimentStatusReport.dat"
                toyo_data2 = pd.read_csv(toyoworkpath2, sep=",", engine="c", encoding="CP949", on_bad_lines='skip')
                toyo_data2 = toyo_data2.iloc[:, [0, 8, 5, 21, 15, 6, 7, 9, 3]]
                toyo_data2.index = toyo_data2.index + 1
                toyo_data2.columns = ['chno', 'use', 'testname', 'folder', 'temp', 'cyc1', 'cyc2', 'cyc3', 'vol']
            toyo_data["chno"] = toyo_data["chno"].astype(int)
            toyo_data["use"] = toyo_data["use"].astype(int)
            toyo_data["day"] = toyo_data['testname'].apply(self.split_value0)
            toyo_data["part"] = toyo_data['testname'].apply(self.split_value1)
            toyo_data["name"] = toyo_data['testname'].apply(self.split_value2)
            toyo_data["path"] = toyo_data['testname']
            if toyo_num != 3:
                toyo_data["folder"] = toyo_data2["folder"]
                toyo_data["temp"] = toyo_data2['temp']
                toyo_data["cyc"] = toyo_data2['cyc1'].astype(str) + " / " + toyo_data2['cyc2'].astype(str) + " / " + toyo_data2['cyc3'].astype(str)
                toyo_data["vol"] = toyo_data2['vol'].where((toyo_data2["vol"].astype('float') < 5) & (toyo_data2["vol"].astype('float') > 2), "-")
            else:
                toyo_data["folder"] = toyo_data['testname'].str.split(" ").str[2]
                toyo_data["temp"] = toyo_data['testname'].str.split(" ").str[2]
                toyo_data["cyc"] = toyo_data['testname'].str.split(" ").str[2]
                toyo_data["vol"] = toyo_data['testname'].str.split(" ").str[2]
            toyo_data["cyclername"] = blkname
            used_chnl = toyo_data["use"].sum()
            toyo_data.loc[(toyo_data["chno"] == 1) & (toyo_data["use"] == 0), "use"] = "완료"
            toyo_data.loc[(toyo_data["chno"] == 0) & (toyo_data["use"] == 0), "use"] = "작업정지"
            toyo_data.loc[toyo_data["use"] == 1, "use"] = "작업중"
            toyo_data["chno"] = toyo_data.index
        return [toyo_data, used_chnl]

    def toyo_data_make(self, toyo_num, blkname):
        toyo_data = self.toyo_base_data_make(toyo_num, blkname)
        self.df = toyo_data[0]
        self.AllchnlData = pd.concat([self.AllchnlData, self.df])

    def toyo_table_make(self, num_i, num_j, toyo_num, blkname):
        toyo_data = self.toyo_base_data_make(toyo_num, blkname)
        self.df = toyo_data[0]
        self.tb_summary.setItem(0, 0, QtWidgets.QTableWidgetItem(str(num_i * num_j - toyo_data[1])))
        self.tb_summary.setItem(1, 0, QtWidgets.QTableWidgetItem(str(toyo_data[1])))
        for i in range(1, num_i + 1):
            for j in range(1, num_j + 1):
                # 첫번째 선택은 채널 번호
                chnl_name = i + (j - 1) * num_i
                column_name = self.toyo_column_list[self.tb_info.currentIndex() + 1]
                self.tb_channel.setItem(j - 1, i - 1, QtWidgets.QTableWidgetItem(str(chnl_name).zfill(3) + "| " + str(
                    self.df.loc[i + (j - 1) * num_i, str(column_name)])))
                self.tb_channel.item(j - 1, i - 1).setFont(QtGui.QFont("Malgun gothic", 9))
                # text가 있는 부분에 대해서 별도 표시 기능 추가
                if self.df.loc[i + (j - 1) * num_i,"use"] == "작업정지" or self.df.loc[i + (j - 1) * num_i,"use"] == "완료":
                    self.tb_channel.item(j - 1, i - 1).setBackground(QtGui.QColor(255,127,0))
                if toyo_num != 3 and self.df.loc[i + (j - 1) * num_i,"vol"] == "-":
                    self.tb_channel.item(j - 1, i - 1).setBackground(QtGui.QColor(200,255,255))
                # 코인셀 구분
                if (toyo_num == 0 and (i + (j - 1) * num_i) < 17) or ((toyo_num == 0 or toyo_num == 1) and
                                                                      ((i + (j - 1) * num_i) > 64) and ((i + (j - 1) * num_i) < 81)):
                    self.tb_channel.item(j - 1, i - 1).setFont(QtGui.QFont("Malgun gothic", 8))
                # text가 있는 부분에 대해서 별도 표시 기능 추가
                if (str(self.FindText.text()) == "") or (str(self.FindText.text()) in self.df.loc[i + (j - 1) * num_i,"testname"]):
                        # 온도별 구분
                        if (toyo_num == 0 and (i + (j - 1) * num_i) > 64):
                            self.tb_channel.item(j - 1, i - 1).setForeground(QtGui.QColor(255,0,0))
                        if (toyo_num == 1 and (i + (j - 1) * num_i) > 64):
                            self.tb_channel.item(j - 1, i - 1).setForeground(QtGui.QColor(0,0,255))
                        if (toyo_num == 2 and (i + (j - 1) * num_i) > 64):
                            self.tb_channel.item(j - 1, i - 1).setForeground(QtGui.QColor(255,0,0))
                        else:
                            self.tb_channel.item(j - 1, i - 1).setForeground(QtGui.QColor(0,0,0))
                else:
                    self.tb_channel.item(j - 1, i - 1).setForeground(QtGui.QColor(175, 175, 175))
        if self.saveok.isChecked():
            save_file_name = filedialog.asksaveasfilename(initialdir="D://", title="Save File Name", defaultextension=".xlsx")
            if save_file_name:
                self.progressBar.setValue(0)
                writer = pd.ExcelWriter(save_file_name, engine="xlsxwriter")
                self.df.to_excel(writer, index=False)
                writer.close()
                self.progressBar.setValue(100)

    def pne_data_make(self, pne_num, blkname):
        # 경로 확인
        if os.path.isdir(self.pne_work_path_list[pne_num]):
            pneworkpath = self.pne_work_path_list[pne_num]+"\\Module_1_channel_info.json"
            pneworkpath2 = self.pne_work_path_list[pne_num]+"\\Module_2_channel_info.json"
        if os.path.isfile(pneworkpath2):
            with open(pneworkpath) as f1:
                js1 = json.loads(f1.read())
            with open(pneworkpath2) as f2:
                js2 = json.loads(f2.read())
            df1 = pd.DataFrame(js1['Channel'])
            df2 = pd.DataFrame(js2['Channel'])
            self.df = pd.concat([df1, df2])
        else:
            with open(pneworkpath) as f1:
                js1 = json.loads(f1.read())
            self.df = pd.DataFrame(js1['Channel'])
        # 데이터 처리
        if os.path.isfile(pneworkpath):
            temp_data = self.df[["Temperature"]]
            temp_data = temp_data.astype('float') * 1000
            temp_data = temp_data.astype('int')
            self.df = self.df[["Ch_No", "State", "Test_Name", "Schedule_Name", "Current_Cycle_Num", "Step_No", "Total_Cycle_Num", "Voltage",
                               "Result_Path"]]
            self.df.columns = self.toyo_column_list[0:4] + ["Current_Cycle_Num", "Step_No", "Total_Cycle_Num", "Voltage", "Result_Path"]
            self.df = self.df.dropna()
            self.df.index = self.df["chno"].astype('int')
            temp_data.index = self.df["chno"].astype('int')
            self.df["day"] = self.df['testname'].apply(self.split_value0)
            self.df["part"] = self.df['testname'].apply(self.split_value1)
            self.df["name"] = self.df['testname'].apply(self.split_value2)
            self.df["temp"] = temp_data
            self.df["Current_Cycle_Num"] = self.df["Current_Cycle_Num"].apply(lambda x: (" " * (4 - len(x))) + x)
            self.df["Step_No"] = self.df["Step_No"].apply(lambda x: (" " * (4 - len(x))) + x)
            self.df["Total_Cycle_Num"] = self.df["Total_Cycle_Num"].apply(lambda x: (" " * (4 - len(x))) + x)
            self.df["cyc"] = self.df["Step_No"] + " / " + self.df["Current_Cycle_Num"] + " / " +  self.df["Total_Cycle_Num"]
            self.df["vol"] = self.df["Voltage"].where(self.df["Voltage"].astype('float') > 0.04, "-")
            self.df["cyclername"] = blkname
            self.df["chno"] = self.df.index
            # 데이터 경로 변경
            self.df = self.change_drive(self.df, self.pne_data_path_list[pne_num])
            self.AllchnlData = pd.concat([self.AllchnlData, self.df])
    
    def pne_table_make(self, num_i, num_j, pne_num, blkname):
        # 경로 확인
        if os.path.isdir(self.pne_work_path_list[pne_num]):
            pneworkpath = self.pne_work_path_list[pne_num]+"\\Module_1_channel_info.json"
            pneworkpath2 = self.pne_work_path_list[pne_num]+"\\Module_2_channel_info.json"
            if os.path.isfile(pneworkpath2):
                with open(pneworkpath) as f1:
                    js1 = json.loads(f1.read())
                with open(pneworkpath2) as f2:
                    js2 = json.loads(f2.read())
                df1 = pd.DataFrame(js1['Channel'])
                df2 = pd.DataFrame(js2['Channel'])
                self.df = pd.concat([df1, df2])
            else:
                with open(pneworkpath) as f1:
                    try:
                        js1 = json.loads(f1.read())
                    except json.JSONDecodeError as e:
                        print(f"JSON 오류: {e} 라인 {e.line} 수정 필요")
                self.df = pd.DataFrame(js1['Channel'])
        # 데이터 처리
            temp_data = self.df[["Temperature"]]
            temp_data = temp_data.astype('float') * 1000
            temp_data = temp_data.astype('int')
            self.df = self.df[["Ch_No", "State", "Test_Name", "Schedule_Name", "Current_Cycle_Num", "Step_No", "Total_Cycle_Num", "Voltage",
                               "Result_Path"]]
            self.df.columns = self.toyo_column_list[0:4] + ["Current_Cycle_Num", "Step_No", "Total_Cycle_Num", "Voltage", "Result_Path"]
            self.df = self.df.dropna()
            self.df.index = self.df["chno"].astype('int')
            temp_data.index = self.df["chno"].astype('int')
            self.df["day"] = self.df['testname'].apply(self.split_value0)
            self.df["part"] = self.df['testname'].apply(self.split_value1)
            self.df["name"] = self.df['testname'].apply(self.split_value2)
            self.df["temp"] = temp_data
            self.df["Current_Cycle_Num"] = self.df["Current_Cycle_Num"].apply(lambda x: (" " * (4 - len(x))) + x)
            self.df["Step_No"] = self.df["Step_No"].apply(lambda x: (" " * (4 - len(x))) + x)
            self.df["Total_Cycle_Num"] = self.df["Total_Cycle_Num"].apply(lambda x: (" " * (4 - len(x))) + x)
            self.df["cyc"] = self.df["Step_No"] + " / " + self.df["Current_Cycle_Num"] + " / " +  self.df["Total_Cycle_Num"]
            self.df["vol"] = self.df["Voltage"].where(self.df["Voltage"].astype('float') > 0.04, "-")
            self.df["cyclername"] = blkname
            self.df["chno"] = self.df.index
            # 데이터 경로 변경
            self.df = self.change_drive(self.df, self.pne_data_path_list[pne_num])
            usedchnlno = len(self.df[(self.df.use =="완료") | (self.df.use == "대기") | (self.df.use == "준비")])
            self.tb_summary.setItem(0, 0, QtWidgets.QTableWidgetItem(str(usedchnlno)))
            self.tb_summary.setItem(1, 0, QtWidgets.QTableWidgetItem(str(num_i * num_j - usedchnlno)))
            for i in range(1, num_i + 1):
                for j in range(1, num_j + 1):
                    chnl_name = i + (j - 1) * num_i
                    column_name = self.toyo_column_list[self.tb_info.currentIndex() + 1]
                    if self.tb_info.currentIndex() == 9:
                        self.tb_channel.setItem(j - 1, i - 1, QtWidgets.QTableWidgetItem(str(self.df.loc[i + (j - 1) * num_i, str(column_name)])))
                        self.tb_channel.item(j - 1, i - 1).setFont(QtGui.QFont("Malgun gothic", 9))
                    else:
                        self.tb_channel.setItem(j - 1, i - 1, QtWidgets.QTableWidgetItem(str(chnl_name).zfill(3) + "| " + str(
                            self.df.loc[i + (j - 1) * num_i, str(column_name)])))
                        self.tb_channel.item(j - 1, i - 1).setFont(QtGui.QFont("Malgun gothic", 9))
                    # 채널 구분 
                    # # 사용 가능 채널 구분 _ 하늘색
                    if (self.df.loc[i + (j - 1) * num_i,"use"] == "대기") or (self.df.loc[i + (j - 1) * num_i,"use"] == "준비"):
                        self.tb_channel.item(j - 1, i - 1).setBackground(QtGui.QColor(200,255,255))
                    elif (self.df.loc[i + (j - 1) * num_i,"use"] == "완료"): # 사용 가능 채널 구분 _ 주황색
                        self.tb_channel.item(j - 1, i - 1).setBackground(QtGui.QColor(255,127,0))
                    elif self.df.loc[i + (j - 1) * num_i,"use"] == "작업멈춤": # 정지 채널 구분 _ 붉은색
                        self.tb_channel.item(j - 1, i - 1).setBackground(QtGui.QColor(255,200,229))
                    # text가 있는 부분에 대해서 별도 표시 기능 추가
                    if (str(self.FindText.text()) == "") or (str(self.FindText.text()) in self.df.loc[i + (j - 1) * num_i,"testname"]):
                            # 온도별 구분
                            if self.df.loc[i + (j - 1) * num_i, "temp"] > 10 and self.df.loc[i + (j - 1) * num_i, "temp"] <= 20:
                                self.tb_channel.item(j - 1, i - 1).setForeground(QtGui.QColor(0,0,255)) # 15도 파란색
                            elif self.df.loc[i + (j - 1) * num_i, "temp"] > 30 and self.df.loc[i + (j - 1) * num_i, "temp"] <= 40:
                                self.tb_channel.item(j - 1, i - 1).setForeground(QtGui.QColor(0,255,0)) # 35도 녹색
                            elif self.df.loc[i + (j - 1) * num_i, "temp"] > 40 and self.df.loc[i + (j - 1) * num_i, "temp"] <= 50:
                                self.tb_channel.item(j - 1, i - 1).setForeground(QtGui.QColor(255,0,0)) # 45도 빨간색
                            else:
                                self.tb_channel.item(j - 1, i - 1).setForeground(QtGui.QColor(0,0,0)) # 기본 검은색
                    else:
                        self.tb_channel.item(j - 1, i - 1).setForeground(QtGui.QColor(175, 175, 175))
            if self.saveok.isChecked():
                save_file_name = filedialog.asksaveasfilename(initialdir="D://", title="Save File Name", defaultextension=".xlsx")
                if save_file_name:
                    self.progressBar.setValue(0)
                    writer = pd.ExcelWriter(save_file_name, engine="xlsxwriter")
                    self.df.to_excel(writer, index=False)
                    writer.close()
                    self.progressBar.setValue(100)

    def table_reset(self):
        self.tb_channel.clear()

    def change_drive(self, df, changed):
        # 상세 데이터부터 범용 데이터 순으로 바꾸기 진행
        original_drive = ["D:\\Data\\", "D:\\DATA\\","D:\\", "E:\\Data\\", "E:\\DATA\\", "E:\\"]
        df["path"] = df["Result_Path"].apply(lambda x: x[:x.rfind('\\')+1])
        for cycler_drive in original_drive:
            df["path"] = df["path"].str.replace(cycler_drive, changed)
        return df

    def cycle_error(self):
        err_msg('파일 or 경로없음!!','C드라이브에 cycler_path.txt 파일이 없거나 toyo/PNE 경로 설정 오류')

    def tb_cycler_combobox(self):
        toyo_table_makers = {
        "Toyo1":(8, 16, 0, self.toyo_cycler_name[0]),
        "Toyo2":(8, 16, 1, self.toyo_cycler_name[1]),
        "Toyo3":(8, 16, 2, self.toyo_cycler_name[2]),
        "Toyo4":(5, 2, 3, self.toyo_cycler_name[3]),
        "Toyo5":(5, 4, 4, self.toyo_cycler_name[4])
        }
        pne_table_makers = {
        "PNE1": (8, 16, 0, self.pne_cycler_name[0]),
        "PNE2": (8, 12, 1, self.pne_cycler_name[1]),
        "PNE3": (8, 4, 2, self.pne_cycler_name[2]),
        "PNE4": (8, 4, 3, self.pne_cycler_name[3]),
        "PNE5": (8, 4, 4, self.pne_cycler_name[4]),
        "PNE01": (8, 4, 5, self.pne_cycler_name[5]),
        "PNE02": (8, 4, 6, self.pne_cycler_name[6]),
        "PNE03": (8, 4, 7, self.pne_cycler_name[7]),
        "PNE04": (8, 8, 8, self.pne_cycler_name[8]),
        "PNE05": (8, 8, 9, self.pne_cycler_name[9]),
        "PNE06": (8, 8, 10, self.pne_cycler_name[10]),
        "PNE07": (8, 8, 11, self.pne_cycler_name[11]),
        "PNE08": (8, 8, 12, self.pne_cycler_name[12]),
        "PNE09": (8, 8, 13, self.pne_cycler_name[13]),
        "PNE10": (8, 8, 14, self.pne_cycler_name[14]),
        "PNE11": (8, 8, 15, self.pne_cycler_name[15]),
        "PNE12": (8, 8, 16, self.pne_cycler_name[16]),
        "PNE13": (8, 8, 17, self.pne_cycler_name[17]),
        "PNE14": (8, 8, 18, self.pne_cycler_name[18]),
        "PNE15": (8, 8, 19, self.pne_cycler_name[19]),
        "PNE16": (8, 8, 20, self.pne_cycler_name[20]),
        "PNE17": (8, 8, 21, self.pne_cycler_name[21]),
        "PNE18": (8, 8, 22, self.pne_cycler_name[22]),
        "PNE19": (8, 8, 23, self.pne_cycler_name[23]),
        "PNE20": (8, 8, 24, self.pne_cycler_name[24]),
        "PNE21": (8, 16, 25, self.pne_cycler_name[25]),
        "PNE22": (8, 16, 26, self.pne_cycler_name[26]),
        "PNE23": (8, 4, 27, self.pne_cycler_name[27]),
        "PNE24": (8, 4, 28, self.pne_cycler_name[28]),
        "PNE25": (8, 4, 29, self.pne_cycler_name[29])
        }
        cycler_text = self.tb_cycler.currentText()
        self.table_reset()
        if cycler_text in toyo_table_makers:
            col_count, row_count, index, name = toyo_table_makers[cycler_text]
            self.toyo_table_make(col_count, row_count, index, name)
        if cycler_text in pne_table_makers:
            col_count, row_count, index, name = pne_table_makers[cycler_text]
            self.pne_table_make(col_count, row_count, index, name)

    def tb_room_combobox(self):
        if self.tb_room.currentIndex() == 0:
            self.tb_cycler.clear()
            self.tb_cycler.addItems(["Toyo1", "Toyo2", "Toyo3", "Toyo4", "Toyo5", "PNE1", "PNE2", "PNE3", "PNE4", "PNE5"])
        elif self.tb_room.currentIndex() == 1:
            self.tb_cycler.clear()
            self.tb_cycler.addItems(["PNE01", "PNE02", "PNE03", "PNE04", "PNE05", "PNE06", "PNE07", "PNE08"])
        elif self.tb_room.currentIndex() == 2:
            self.tb_cycler.clear()
            self.tb_cycler.addItems(["PNE09", "PNE10", "PNE11", "PNE12", "PNE13", "PNE14", "PNE15", "PNE16"])
        elif self.tb_room.currentIndex() == 3:
            self.tb_cycler.clear()
            self.tb_cycler.addItems(["PNE17", "PNE18", "PNE19", "PNE20", "PNE21", "PNE22", "PNE23", "PNE24", "PNE25"])
        elif self.tb_room.currentIndex() == 4:
            self.tb_cycler.clear()
            self.tb_cycler.addItems(["Toyo1", "Toyo2", "Toyo3", "Toyo4", "Toyo5", "PNE1", "PNE2", "PNE3", "PNE4", "PNE5",
                                     "PNE01", "PNE02", "PNE03", "PNE04", "PNE05", "PNE06", "PNE07", "PNE08",
                                     "PNE09", "PNE10", "PNE11", "PNE12", "PNE13", "PNE14", "PNE15", "PNE16",
                                     "PNE17", "PNE18", "PNE19", "PNE20", "PNE21", "PNE22", "PNE23", "PNE24", "PNE25"])

    def tb_info_combobox(self):
        self.tb_cycler_combobox()

    def bm_set_profile_button(self):
        self.BMset_battery_status_log_Profile.setDisabled(True)
        global writer
        root = Tk()
        root.withdraw()
        datafilepath = filedialog.askopenfilenames(initialdir="d://", title="Choose Test files")
        self.BMset_battery_status_log_Profile.setEnabled(True)
        if datafilepath:
            filecount = 0
            mincapa = int(self.SetMincapacity.text())
            fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8), (ax9, ax10)) \
                = plt.subplots(nrows=5, ncols=2, figsize=(12, 10))
            if self.saveok.isChecked():
                save_file_name = filedialog.asksaveasfilename(initialdir="D://", title="Save File Name", defaultextension=".xlsx")
                if save_file_name:
                    writer = pd.ExcelWriter(save_file_name, engine="xlsxwriter")
                dfchg = pd.DataFrame()
                dfdchg = pd.DataFrame()
            for filepath in datafilepath:
                filecountmax = len(datafilepath)
                progressdata = filecount/filecountmax * 100
                filecount = filecount + 1
                self.progressBar.setValue(int(progressdata))
                    
                df = pd.read_csv(filepath, usecols=[0, 2, 21, 25, 27, 28], on_bad_lines='skip')
                df.columns = ['Time', 'Curr', 'SOC2', 'SOC', 'Vol', 'Temp']
                if "방전" in filepath:
                    df["Type"] = "Discharge"
                    df["State"] = "Unplugged"
                else:
                    df["Type"] = "Charging"
                    df["State"] = "AC"
                raw_file_split = filepath.split("_")
                CycNo = raw_file_split[1]
                if "csv" in filepath:
                    CycNo = CycNo.replace('.csv','')
                else:
                    CycNo = CycNo.replace('.txt','')
                df["Cyc"] = int(CycNo)
                df=df[df.loc[:,'Curr']!="Batterycurrent"]
                df = df.reset_index()
                df['Time'] = df.index * 2 / 3600
                df['Curr']=df['Curr'].apply(float)/mincapa*(-1)
                df['SOC2']=df['SOC2'].apply(float)/10/mincapa/2
                df['SOC']=df['SOC'].apply(float)
                df['Vol']=df['Vol'].apply(float)/1000
                df['Temp']=df['Temp'].apply(float)/10
                if "방전" in filepath:
                    graph_set_profile(df.Time, df.Vol, ax2, 3.4, 4.8, 0.2, "Time(hr)", "Voltage (V)", "", 0, 0, 11, 1)
                    graph_set_profile(df.Time, df.Curr, ax4, -0.6, 0.1, 0.1, "Time(hr)", "Curr", "", 0, 0, 11, 1)
                    graph_set_profile(df.Time, df.Temp, ax6, 20, 50, 4, "Time(hr)", "temp.", "", 0, 0, 11, 1)
                    graph_set_profile(df.Time, df.SOC, ax8, 0, 120, 10, "Time(hr)", "SOC", "", 0, 0, 11, 1)
                    graph_set_profile(df.Time, df.SOC2, ax10, 0, 110, 10, "Time(hr)", "real SOC", "", 0, 0, 11, 1)
                else:
                    df = df[(df["Time"] < 4)]
                    graph_set_profile(df.Time, df.Vol, ax1, 3.4, 4.8, 0.2, "Time(hr)", "Voltage (V)", "", 0, 0, 4, 1)
                    graph_set_profile(df.Time, df.Curr, ax3, 0, 12, 1.0, "Time(hr)", "Curr", "", 0, 0, 4, 1)
                    graph_set_profile(df.Time, df.Temp, ax5, 20, 50, 4, "Time(hr)", "temp.", "", 0, 0, 4, 1)
                    graph_set_profile(df.Time, df.SOC, ax7, 0, 120, 10, "Time(hr)", "SOC", "", 0, 0, 4, 1)
                    graph_set_profile(df.Time, df.SOC2, ax9, 0, 120, 10, "Time(hr)", "real SOC", CycNo, 0, 0 , 4, 1)
                if self.saveok.isChecked() and save_file_name:
                    if "방전" in filepath:
                    # 방전 Profile 추출용
                        dfdchg = dfdchg._append(df)
                        # dfdchg.to_excel(writer, sheet_name="dchg")
                    else:
                    # 충전 Profile 추출용
                        dfchg = dfchg._append(df)
                        # dfchg.to_excel(writer, sheet_name="chg")
            if self.saveok.isChecked() and save_file_name:
                dfdchg.to_excel(writer, sheet_name="dchg")
                dfchg.to_excel(writer, sheet_name="chg")
                writer.close()
                
            self.progressBar.setValue(100)
            fig.legend()
            plt.tight_layout(pad=1, w_pad=1, h_pad=1)
            plt.subplots_adjust(right=0.8)
            plt.show()
    
    def bm_set_cycle_button(self):
        global writer
        root = Tk()
        root.withdraw()
        setxscale = int(self.setcyclexscale.text())
        datafilepath = filedialog.askdirectory(initialdir="d://", title="Choose Test files")
        if datafilepath:
            fig, ((ax1), (ax2)) = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
            filecount = 0
            mincapa = int(self.SetMincapacity.text())
            if self.saveok.isChecked():
                save_file_name = filedialog.asksaveasfilename(initialdir="D://", title="Save File Name", defaultextension=".xlsx")
                if save_file_name:
                    writer = pd.ExcelWriter(save_file_name, engine="xlsxwriter")
            dfcyc = pd.DataFrame()
            dfcyc2 = pd.DataFrame()
            subfile = [f for f in os.listdir(datafilepath) if f.startswith('방전_')]
            for filepath in subfile:
                filecountmax = len(subfile)
                progressdata = filecount/filecountmax * 100
                filecount = filecount + 1
                self.progressBar.setValue(int(progressdata))
                df = pd.read_csv(datafilepath+"/"+filepath, usecols=[21], on_bad_lines='skip')
                df.columns = ['SOC']
                raw_file_split = filepath.split("_")
                CycNo = raw_file_split[1]
                if "csv" in filepath:
                    CycNo = CycNo.replace('.csv','')
                else:
                    CycNo = CycNo.replace('.txt','')
                df["Cyc"] = int(CycNo)
                df=df[df.loc[:,'SOC']!="Charge_counter"]
                df['SOC']=df['SOC'].apply(float)/10/mincapa/2/100
                dfcyc = dfcyc._append(df.loc[0])
            subfile2 = [f for f in os.listdir(datafilepath) if f.startswith('충전_')]
            for filepath2 in subfile2:
                filecountmax = len(subfile2)
                progressdata = filecount/filecountmax * 100
                filecount = filecount + 1
                self.progressBar.setValue(int(progressdata))
                df2 = pd.read_csv(datafilepath+"/"+filepath2, usecols=[21], on_bad_lines='skip')
                df2.columns = ['SOC2']
                raw_file_split = filepath2.split("_")
                CycNo = raw_file_split[1]
                if "csv" in filepath:
                    CycNo = CycNo.replace('.csv','')
                else:
                    CycNo = CycNo.replace('.txt','')
                df2["Cyc2"] = int(CycNo)
                df2=df2[df2.loc[:,'SOC2']!="Charge_counter"]
                df2['SOC2']=df2['SOC2'].apply(float)/10/mincapa/2/100
                dfcyc2 = dfcyc2._append(df2.iloc[-1])
            dfcyc = dfcyc.sort_values(by="Cyc")
            dfcyc2 = dfcyc2.sort_values(by="Cyc2")
            graph_cycle(dfcyc.Cyc, dfcyc.SOC, ax1, 0.8, 1.05, 0.05, "Cycle", "Discharge Capacity Ratio", datafilepath, setxscale, 0)
            graph_cycle(dfcyc2.Cyc2, dfcyc2.SOC2, ax2, 0.8, 1.05, 0.05, "Cycle", "Charge Capacity Ratio", datafilepath, setxscale, 0)
            if self.saveok.isChecked() and save_file_name:
                dfcyc=dfcyc[["Cyc", "SOC"]]
                dfcyc2=dfcyc2[["Cyc2", "SOC2"]]
                dfcyc = dfcyc.reset_index()
                dfcyc2 = dfcyc2.reset_index()
                dfcyc.to_excel(writer, sheet_name="dchgcyc")
                dfcyc2.to_excel(writer, sheet_name="chgcyc")
                writer.close()
            self.progressBar.setValue(100)
            fig.legend()
            plt.tight_layout(pad=1, w_pad=1, h_pad=1)
            plt.show()
    
    def bm_set_profile_button(self):
        global writer
        root = Tk()
        root.withdraw()
        datafilepath = filedialog.askopenfilenames(initialdir="d://", title="Choose Test files")
        filecount = 0
        mincapa = int(self.SetMincapacity.text())
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8), (ax9, ax10)) \
            = plt.subplots(nrows=5, ncols=2, figsize=(12, 10))
        if self.saveok.isChecked():
            save_file_name = filedialog.asksaveasfilename(initialdir="D://", title="Save File Name", defaultextension=".xlsx")
            if save_file_name:
                writer = pd.ExcelWriter(save_file_name, engine="xlsxwriter")
            dfchg = pd.DataFrame()
            dfdchg = pd.DataFrame()
        for filepath in datafilepath:
            filecountmax = len(datafilepath)
            progressdata = filecount/filecountmax * 100
            filecount = filecount + 1
            self.progressBar.setValue(int(progressdata))
            df = pd.read_csv(filepath, usecols=[0, 2, 21, 25, 27, 28], on_bad_lines='skip')
            df.columns = ['Time', 'Curr', 'SOC2', 'SOC', 'Vol', 'Temp']
            if "방전" in filepath:
                df["Type"] = "Discharge"
                df["State"] = "Unplugged"
            else:
                df["Type"] = "Charging"
                df["State"] = "AC"
            raw_file_split = filepath.split("_")
            CycNo = raw_file_split[1]
            if "csv" in filepath:
                CycNo = CycNo.replace('.csv','')
            else:
                CycNo = CycNo.replace('.txt','')
            df["Cyc"] = int(CycNo)
            df=df[df.loc[:,'Curr']!="Batterycurrent"]
            df = df.reset_index()
            df['Time'] = df.index * 2 / 3600
            df['Curr']=df['Curr'].apply(float)/mincapa*(-1)
            df['SOC2']=df['SOC2'].apply(float)/10/mincapa/2
            df['SOC']=df['SOC'].apply(float)
            df['Vol']=df['Vol'].apply(float)/1000
            df['Temp']=df['Temp'].apply(float)/10
            if "방전" in filepath:
                graph_set_profile(df.Time, df.Vol, ax2, 3.4, 4.8, 0.2, "Time(hr)", "Voltage (V)", "", 0, 0, 11, 1)
                graph_set_profile(df.Time, df.Curr, ax4, -0.6, 0.1, 0.1, "Time(hr)", "Curr", "", 0, 0, 11, 1)
                graph_set_profile(df.Time, df.Temp, ax6, 20, 50, 4, "Time(hr)", "temp.", "", 0, 0, 11, 1)
                graph_set_profile(df.Time, df.SOC, ax8, 0, 120, 10, "Time(hr)", "SOC", "", 0, 0, 11, 1)
                graph_set_profile(df.Time, df.SOC2, ax10, 0, 110, 10, "Time(hr)", "real SOC", "", 0, 0, 11, 1)
            else:
                df = df[(df["Time"] < 4)]
                graph_set_profile(df.Time, df.Vol, ax1, 3.4, 4.8, 0.2, "Time(hr)", "Voltage (V)", "", 0, 0, 4, 1)
                graph_set_profile(df.Time, df.Curr, ax3, 0, 12, 1.0, "Time(hr)", "Curr", "", 0, 0, 4, 1)
                graph_set_profile(df.Time, df.Temp, ax5, 20, 50, 4, "Time(hr)", "temp.", "", 0, 0, 4, 1)
                graph_set_profile(df.Time, df.SOC, ax7, 0, 120, 10, "Time(hr)", "SOC", "", 0, 0, 4, 1)
                graph_set_profile(df.Time, df.SOC2, ax9, 0, 120, 10, "Time(hr)", "real SOC", CycNo, 0, 0, 4, 1)
            if self.saveok.isChecked() and save_file_name:
                if "방전" in filepath:
                    dfdchg = dfdchg._append(df)
                else:
                    dfchg = dfchg._append(df)
        if self.saveok.isChecked() and save_file_name:
            dfdchg.to_excel(writer, sheet_name="dchg")
            dfchg.to_excel(writer, sheet_name="chg")
            writer.close()
        fig.legend()
        plt.tight_layout(pad=1, w_pad=1, h_pad=1)
        plt.subplots_adjust(right=0.8)
        self.progressBar.setValue(100)
        plt.show()
    
    def bm_set_cycle_button(self):
        self.BMSetCycle.setDisabled(True)
        global writer
        root = Tk()
        root.withdraw()
        setxscale = int(self.setcyclexscale.text())
        datafilepath = filedialog.askdirectory(initialdir="d://", title="Choose Test files")
        self.BMSetCycle.setEnabled(True)
        fig, ((ax1), (ax2)) = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
        filecount = 0
        mincapa = int(self.SetMincapacity.text())
        if self.saveok.isChecked():
            save_file_name = filedialog.asksaveasfilename(initialdir="D://", title="Save File Name", defaultextension=".xlsx")
            if save_file_name:
                writer = pd.ExcelWriter(save_file_name, engine="xlsxwriter")
        dfcyc = pd.DataFrame()
        dfcyc2 = pd.DataFrame()
        subfile = [f for f in os.listdir(datafilepath) if f.startswith('방전_')]
        for filepath in subfile:
            filecountmax = len(subfile)
            progressdata = filecount/filecountmax * 100
            filecount = filecount + 1
            self.progressBar.setValue(int(progressdata))
            df = pd.read_csv(datafilepath+"/"+filepath, usecols=[21], on_bad_lines='skip')
            df.columns = ['SOC']
            raw_file_split = filepath.split("_")
            CycNo = raw_file_split[1]
            if "csv" in filepath:
                CycNo = CycNo.replace('.csv','')
            else:
                CycNo = CycNo.replace('.txt','')
            df["Cyc"] = int(CycNo)
            df=df[df.loc[:,'SOC']!="Charge_counter"]
            df['SOC']=df['SOC'].apply(float)/10/mincapa/2/100
            dfcyc = dfcyc._append(df.loc[0])
        subfile2 = [f for f in os.listdir(datafilepath) if f.startswith('충전_')]
        for filepath2 in subfile2:
            filecountmax = len(subfile2)
            progressdata = filecount/filecountmax * 100
            filecount = filecount + 1
            self.progressBar.setValue(int(progressdata))
            df2 = pd.read_csv(datafilepath+"/"+filepath2, usecols=[21], on_bad_lines='skip')
            df2.columns = ['SOC2']
            raw_file_split = filepath2.split("_")
            CycNo = raw_file_split[1]
            if "csv" in filepath:
                CycNo = CycNo.replace('.csv','')
            else:
                CycNo = CycNo.replace('.txt','')
            df2["Cyc2"] = int(CycNo)
            df2=df2[df2.loc[:,'SOC2']!="Charge_counter"]
            df2['SOC2']=df2['SOC2'].apply(float)/10/mincapa/2/100
            dfcyc2 = dfcyc2._append(df2.iloc[-1])
        dfcyc = dfcyc.sort_values(by="Cyc")
        dfcyc2 = dfcyc2.sort_values(by="Cyc2")
        graph_cycle(dfcyc.Cyc, dfcyc.SOC, ax1, 0.7, 1.05, 0.05, "Cycle", "Discharge Capacity Ratio", datafilepath, setxscale, 0)
        graph_cycle(dfcyc2.Cyc2, dfcyc2.SOC2, ax2, 0.7, 1.05, 0.05, "Cycle", "Charge Capacity Ratio", datafilepath, setxscale, 0)
        if self.saveok.isChecked() and save_file_name:
            dfcyc=dfcyc[["Cyc", "SOC"]]
            dfcyc2=dfcyc2[["Cyc2", "SOC2"]]
            dfcyc = dfcyc.reset_index()
            dfcyc2 = dfcyc2.reset_index()
            dfcyc.to_excel(writer, sheet_name="dchgcyc")
            dfcyc2.to_excel(writer, sheet_name="chgcyc")
            writer.close()
        self.progressBar.setValue(100)
        fig.legend()
        plt.tight_layout(pad=1, w_pad=1, h_pad=1)
        plt.show()

    def battery_dump_data(self, battery_dump_path):
        # 0:Time, 1:VOLTAGE NOW, 2:CURRENT NOW, 3:CURRENT MAX, 4:CHARGING CURRENT, 5:CAPACITY, 6:TBAT, 7:TUSB, 8:TCHG, 9:TWPC, 10:TBLK,
        # 11:TLRP, 12:TDCHG, 13:TSUB, 14:BATTERY STATUS, 15:DIRECT CHARGER STATUS, 16:CHARGING MODE, 17:HEALTH STATUS, 18:CABLE TYPE, 19:MUIC CABLE TYPE,
        # 20:THERMAL ZONE, 21:SLATE MODE, 22:STORE MODE, 23:SAFETY TIMER, 24:CURRENT EVENT, 25:MISC EVENT, 26:TX EVENT, 27:PHM, 28:SRCCAP_TRANSIT,
        # 29:SYS AVG CURRENT, 30:BD_VERSION, 31:VID, 32:PID, 33:XID, 34:VOLTAGE PACK MAIN, 35:VOLTAGE PACK SUB, 36:CURRENT NOW MAIN, 37:CURRENT NOW SUB,
        # 38:CYCLE, 39:OCV, 40:RAW SOC, 41:CAPACITY MAX, 42:WRL_MODE, 43:TX VOUT, 44:TX IOUT, 45:PING FRQ, 46:MIN OP FRQ, 47:MAX OP FRQ, 48:PHM,
        # 49:RX TYPE, 50:OTP FMWR VERSION, 51:WC IC REV
        
        #1:DATE TIME  	2:VOLTAGE NOW 	3:CURRENT NOW 	4:CURRENT MAX 	5:CHARGING CURRENT 	6:CAPACITY 	7:TBAT 	8:TUSB 	9:TCHG 	10:TWPC
        #11:TBLK 	12:TLRP 	13:TDCHG 	14:TSUB 	15:BATTERY STATUS 	16:DIRECT CHARGER STATUS 	17:CHARGING MODE 	
        #18:HEALTH STATUS 	19:CABLE TYPE 	20:MUIC CABLE TYPE 	21:THERMAL ZONE 	22:SLATE MODE 	23:STORE MODE 	24:SAFETY TIMER
        #5:CURRENT EVENT 	26:MISC EVENT 	27:TX EVENT 	28:PHM 	29:SRCCAP TRANSIT 	30:SYS AVG CURRENT 	31:BD_VERSION 	
        #32:DC_RATIO 	33:C_PDO/A_PDO-MAX_V-MIN_V-MAX_CUR 	34:VID 	35:PID 	36:XID 	37:VOLTAGE PACK MAIN 	38:CURRENT NOW MAIN
        #39:CYCLE 	40:OCV 	41:RAW SOC 	42:CAPACITY MAX 	43:WRL_MODE 

        batterydump1 = pd.read_csv(battery_dump_path + "//battery_dump1", sep=",", on_bad_lines='skip')
        batterydump2 = pd.read_csv(battery_dump_path + "//battery_dump2", sep=",", on_bad_lines='skip')
        batterydump1 = batterydump1.iloc[:, [0, 1, 2, 5, 6, 7, 8, 11, 14, 15, 29, 38, 39, 40, 41]]
        batterydump2 = batterydump2.iloc[:, [0, 1, 2, 5, 6, 7, 8, 11, 14, 15, 29, 38, 39, 40, 41]]
        batterydump1.columns = ['Time_temp', 'Vol', 'Curr', 'SOC', 'T_bat', 'T_usb', 'T_chg', 'T_lrp', 'battery_status', 'direct_charger_status',
                               'sys_avg_current', 'cycle', 'ocv', 'SOCraw', 'cap_max']
        batterydump2.columns = ['Time_temp', 'Vol', 'Curr', 'SOC', 'T_bat', 'T_usb', 'T_chg', 'T_lrp', 'battery_status', 'direct_charger_status',
                               'sys_avg_current', 'cycle', 'ocv', 'SOCraw', 'cap_max']
        Profile = pd.concat([batterydump1, batterydump2], axis=0).reset_index(drop=True)
        Profile = Profile.dropna()
        # batterydump1 = batterydump1.iloc[:, [0, 1, 2, 5, 6, 7, 8, 11, 14, 15, 29, 38, 39, 40, 41]]
        # batterydump1.columns = ['Time_temp', 'Vol', 'Curr', 'SOC', 'T_bat', 'T_usb', 'T_chg', 'T_lrp', 'battery_status', 'direct_charger_status',
        #                        'sys_avg_current', 'cycle', 'ocv', 'SOCraw', 'capacity_max']
        # Profile = batterydump.iloc[:, [0, 1, 2, 5, 6, 7, 8, 11, 14, 15, 29, 38, 39, 40, 41]]
        # Profile.columns = ['Time_temp', 'Vol', 'Curr', 'SOC', 'T_bat', 'T_usb', 'T_chg', 'T_lrp', 'battery_status', 'direct_charger_status',
        #                        'sys_avg_current', 'cycle', 'ocv', 'SOCraw', 'capacity_max']
        # 2023-09-10 20:08:25+0530
        Profile["Time"] = Profile["Time_temp"].str[:-5]
        Profile["Time"] = pd.to_datetime(Profile["Time"], format= '%Y-%m-%d %H:%M:%S')
        Profile["Time"] = Profile["Time"] - Profile["Time"][0]
        Profile["Time"] = Profile["Time"].dt.total_seconds().div(3600).astype(float)
        Profile = Profile.loc[Profile["Time"] > -480000]
        Profile["Time"] = Profile["Time"] - Profile["Time"].nsmallest(100).iloc[-1]
        # max_limit = Profile["Time"].nlargest(1).iloc[-1]
        Profile = Profile.loc[(Profile["Time"] > 0)]
        Profile['Curr']=Profile['Curr'].apply(float)/1000
        Profile['SOC']=Profile['SOC'].apply(float)
        Profile['Vol']=Profile['Vol'].apply(float)/1000
        Profile['ocv']=Profile['ocv'].apply(float)/1000
        Profile['T_bat']=Profile['T_bat'].apply(float)/10
        Profile['T_usb']=Profile['T_usb'].apply(float)/10
        Profile['T_chg']=Profile['T_chg'].apply(float)/10
        Profile['T_lrp']=Profile['T_lrp'].apply(float)/10
        Profile['SOCraw']=Profile['SOCraw'].apply(float)/100
        Profile['cap_max']=Profile['cap_max'].apply(float)/10
        # Profile['capacity_max']=Profile['capacity_max'].apply(float)/10
        return Profile

    def set_tab_reset_button(self):
        self.tab_delete(self.set_tab)
        self.tab_no = 0
        
    def set_log_confirm_button(self):
        # battery_dump profile
        # 0:Time, 1:VOLTAGE NOW, 2:CURRENT NOW, 3:CURRENT MAX, 4:CHARGING CURRENT, 5:CAPACITY, 6:TBAT, 7:TUSB, 8:TCHG, 9:TWPC, 10:TBLK,
        # 11:TLRP, 12:TDCHG, 13:TSUB, 14:BATTERY STATUS, 15:DIRECT CHARGER STATUS, 16:CHARGING MODE, 17:HEALTH STATUS, 18:CABLE TYPE, 19:MUIC CABLE TYPE,
        # 20:THERMAL ZONE, 21:SLATE MODE, 22:STORE MODE, 23:SAFETY TIMER, 24:CURRENT EVENT, 25:MISC EVENT, 26:TX EVENT, 27:PHM, 28:SRCCAP_TRANSIT,
        # 29:SYS AVG CURRENT, 30:BD_VERSION, 31:VID, 32:PID, 33:XID, 34:VOLTAGE PACK MAIN, 35:VOLTAGE PACK SUB, 36:CURRENT NOW MAIN, 37:CURRENT NOW SUB,
        # 38:CYCLE, 39:OCV, 40:RAW SOC, 41:CAPACITY MAX, 42:WRL_MODE, 43:TX VOUT, 44:TX IOUT, 45:PING FRQ, 46:MIN OP FRQ, 47:MAX OP FRQ, 48:PHM,
        # 49:RX TYPE, 50:OTP FMWR VERSION, 51:WC IC REV
        global writer
        root = Tk()
        root.withdraw()
        self.SetlogConfirm.setDisabled(True)
        datafilepath = filedialog.askdirectory(initialdir="d://", title="Choose Test Folders")
        self.SetlogConfirm.setEnabled(True)
        # self.tab_delete(self.set_tab)
        if datafilepath:
        # 최근 사이클 산정 및 전체 사이클 적용여부 확인
            if self.saveok.isChecked():
                save_file_name = filedialog.asksaveasfilename(initialdir="D://", title="Save File Name", defaultextension=".xlsx")
                if save_file_name:
                    writer = pd.ExcelWriter(save_file_name, engine="xlsxwriter")
            fig, ax = plt.subplots(nrows=5, ncols=1, figsize=(6, 10))
            tab = QtWidgets.QWidget()
            tab_layout = QtWidgets.QVBoxLayout(tab)
            canvas = FigureCanvas(fig)
            toolbar = NavigationToolbar(canvas, None)
            Profile = self.battery_dump_data(datafilepath)
            self.SetMaxCycle.setText(str(Profile.cycle.max()))
        #Short Profile 확인용
            graph_set_profile(Profile.Time, Profile.Vol, ax[0], 3.0, 4.8, 0.2, "Time(hr)", "Volt.(V)", "CCV", 1, 0, 0, 0)
            graph_set_profile(Profile.Time, Profile.ocv, ax[0], 3.0, 4.8, 0.2, "Time(hr)", "Volt.(V)", "OCV", 2, 0, 0, 0)
            graph_set_profile(Profile.Time, Profile.Curr, ax[1], -5, 6, 1, "Time(hr)", "Curr.", "Curr", 1, 0, 0, 0)
            graph_set_profile(Profile.Time, Profile.SOC, ax[2], 0, 120, 10, "Time(hr)", "SOC", "SOC", 1, 0, 0, 0)
            graph_set_profile(Profile.Time, Profile.SOCraw, ax[2], 0, 120, 10, "Time(hr)", "SOC", "rawSOC", 2, 0, 0, 0)
            graph_set_profile(Profile.Time, Profile.T_lrp, ax[3], 20, 50, 4, "Time(hr)", "temp.", "T_lrp", 4, 0, 0, 0)
            graph_set_profile(Profile.Time, Profile.T_usb, ax[3], 20, 50, 4, "Time(hr)", "temp.", "T_usb", 2, 0, 0, 0)
            graph_set_profile(Profile.Time, Profile.T_chg, ax[3], 20, 50, 4, "Time(hr)", "temp.", "T_chg", 3, 0, 0, 0)
            graph_set_profile(Profile.Time, Profile.T_bat, ax[3], 20, 50, 4, "Time(hr)", "Temp.(℃)", "T_bat", 1, 0, 0, 0)
            graph_set_profile(Profile.Time, Profile.cap_max, ax[4], 90, 101, 1, "Time(hr)", "Cap_max", "cap_max", 1, 0, 0, 0)
        # Short 관련
            ax[0].legend(loc="lower left")
            ax[1].legend(loc="lower left")
            ax[2].legend(loc="lower left")
            ax[3].legend(loc="lower left")
            ax[4].legend(loc="lower left")
            for i in range(4):
                # X축 레이블 제거
                ax[i].set_xlabel('')
                # X축 틱 레이블 제거
                ax[i].set_xticklabels([])
            Chgnamelist = datafilepath.split("/")
            tab_layout.addWidget(toolbar)
            tab_layout.addWidget(canvas)
            self.set_tab.addTab(tab, Chgnamelist[-1])
            self.set_tab.setCurrentWidget(tab)
            self.tab_no = self.tab_no + 1
            plt.tight_layout(pad=1, w_pad=1, h_pad=1)
        if self.saveok.isChecked() and save_file_name:
            Profile.to_excel(writer)
            writer.close()
        plt.tight_layout(pad=1, w_pad=1, h_pad=1)
        plt.close()
        self.progressBar.setValue(100)

    # def SetlogConfirmbutton(self):
    #     '''
    #     Act program Set log
    #     0:[TIME] 1: IMEI 2: Binary version 3: Capacity 4: cisd_fullcaprep_max 5: batt_charging_source
    #     6: charging_type 7: voltage_now 8: voltage_avg 9: current_now 10: current_avg
    #     11: battery_temp 12: ac_temp 13: temperature 14: battery_cycle 15: battery_charger_status
    #     16: batt_slate_mode 17: fg_asoc 18: fg_cycle 19: BIG 20: Little
    #     21: G3D 22: ISP 23: curr_5 24: wc_vrect 25: wc_vout
    #     26: dchg_temp 27: dchg_temp_adc 28: direct_charging_iin 29: AP CUR_CH0 30: AP CUR_CH1
    #     31: AP CUR_CH2 32: AP CUR_CH3 33: AP CUR_CH4 34: AP CUR_CH5 35: AP CUR_CH6
    #     36: AP CUR_CH7 37: AP POW_CH0 38: AP POW_CH1 39: AP POW_CH2 40: AP POW_CH3
    #     41: AP POW_CH4 42: AP POW_CH5 43: AP POW_CH6 44: AP POW_CH7 45: cisd_data
    #     46: LRP 47: USB_TEMP
    #     '''
    #     global writer
    #     root = Tk()
    #     root.withdraw()
    #     self.SetlogConfirm.setDisabled(True)
    #     datafilepath = filedialog.askopenfilenames(initialdir="d://", title="Choose Test files")
    #     self.SetlogConfirm.setEnabled(True)
    #     if datafilepath:
    #         recentcycno = int(self.recentcycleno.text())
    #         filecount = 0
    #         if self.saveok.isChecked():
    #             save_file_name = filedialog.asksaveasfilename(initialdir="D://", title="Save File Name", defaultextension=".xlsx")
    #             if save_file_name:
    #                 writer = pd.ExcelWriter(save_file_name, engine="xlsxwriter")
    #             cycoutputdf = pd.DataFrame({'name': [''], 'cycle': [''], 'Chg_realSOC': [''], 'Dchg_realSOC': [''], 'Chg time(min)': [''], 'Dchg time(min)': ['']})
    #             chgoutputdf = pd.DataFrame()
    #             dchgoutputdf = pd.DataFrame()
    #         for filepath in datafilepath:
    #             mincapa = int(self.SetMincapacity.text())
    #             fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8), (ax9, ax10)) \
    #                 = plt.subplots(nrows=5, ncols=2, figsize=(12, 10))
    #             chkcyc = set_log_cycle(filepath, self.realcyc.isChecked(), recentcycno, self.allcycle.isChecked(),
    #                                    self.manualcycle.isChecked(), self.manualcycleno) 
    #             filecountmax = len(datafilepath)
    #             filecount = filecount + 1
    #             # 전체 사이클과 최근 사이클 기준 설정
    #             if self.allcycle.isChecked() == True:
    #                 cyclecountmax = range(chkcyc[0], chkcyc[1] + 1)
    #             elif self.manualcycle.isChecked() == True:
    #                 manualcyclenochk = list(map(int, (self.manualcycleno.text().split())))
    #                 if len(manualcyclenochk) > 2:
    #                     manualcyclenochk = [x for x in manualcyclenochk if (x >= chkcyc[0] and x <= chkcyc[1])]
    #                     cyclecountmax = manualcyclenochk
    #                 else:
    #                     cycmin = max(chkcyc[0], manualcyclenochk[0])
    #                     cycmax = min(chkcyc[1], manualcyclenochk[1])
    #                     cyclecountmax = range(cycmin, cycmax + 1)
    #             else:
    #             # 최근 20 cycle 기준으로 설정
    #                 if (chkcyc[1] - chkcyc[0]) > recentcycno:
    #                     cyclecountmax = range(chkcyc[1] - recentcycno , chkcyc[1] + 1)
    #                 else:
    #                     cyclecountmax = range(chkcyc[0], chkcyc[1] + 1)
    #             namelist = filepath.split("/")
    #             for i in cyclecountmax:
    #                 temp = set_act_log_Profile(chkcyc[2].Profile, mincapa, i)
    #                 progressdata = ((filecount - 1) + (i - cyclecountmax[0] + 1)/len(cyclecountmax))/filecountmax * 100
    #                 self.progressBar.setValue(int(progressdata))
    #                 if hasattr(temp, "ChgProfile"):
    #                     chgrealcap = str(round(temp.ChgProfile.SOC2.max(), 2))
    #                     chgmaxtime = str(round(temp.ChgProfile.Time.max() * 60, 1))
    #                     caplegend = "{:3}".format(str(i)) + " C:" + "{:8}".format(chgrealcap)
    #                     temp.ChgProfile = temp.ChgProfile[(temp.ChgProfile["Time"] < 4)]
    #                     graph_set(temp.ChgProfile.Time, temp.ChgProfile.Vol, ax1, 3.0, 4.8, 0.2, "Time(hr)", "Voltage (V)", "", 0)
    #                     graph_set(temp.ChgProfile.Time, temp.ChgProfile.Curr, ax3, 0, 2.4, 0.2, "Time(hr)", "Curr", "", 0)
    #                     graph_set(temp.ChgProfile.Time, temp.ChgProfile.Temp, ax5, 20, 50, 4, "Time(hr)", "temp.", "", 0)
    #                     graph_set(temp.ChgProfile.Time, temp.ChgProfile.SOC, ax7, 0, 120, 10, "Time(hr)", "SOC", "", 0)
    #                     graph_set(temp.ChgProfile.Time, temp.ChgProfile.SOC2, ax9, 0, 120, 10, "Time(hr)", "real SOC", "", 0)
    #                 else:
    #                     chgrealcap = " "
    #                     chgmaxtime = " "
    #                 if hasattr(temp, "DchgProfile"):
    #                     dchgrealcap = str(round(temp.DchgProfile.SOC2.max(), 2))
    #                     dchgmaxtime = str(round(temp.DchgProfile.Time.max() * 60, 1))
    #                     caplegend = caplegend + " D:" + "{:6}".format(dchgrealcap)
    #                     graph_set(temp.DchgProfile.Time, temp.DchgProfile.Vol, ax2, 3.0, 4.6, 0.2, "Time(hr)", "Voltage (V)", "", 1)
    #                     graph_set(temp.DchgProfile.Time, temp.DchgProfile.Curr, ax4, -0.6, 0.1, 0.1, "Time(hr)", "Curr", "", 1)
    #                     graph_set(temp.DchgProfile.Time, temp.DchgProfile.Temp, ax6, 20, 50, 4, "Time(hr)", "temp.", "", 1)
    #                     graph_set(temp.DchgProfile.Time, temp.DchgProfile.SOC, ax8, 0, 120, 10, "Time(hr)", "SOC", "", 1)
    #                     if self.saveok.isChecked():
    #                         graph_set(temp.DchgProfile.Time, 100 - temp.DchgProfile.SOC2, ax10, 0, 120, 10, "Time(hr)", "real SOC", i, 1)
    #                     else:
    #                         graph_set(temp.DchgProfile.Time, 100 - temp.DchgProfile.SOC2, ax10, 0, 120, 10, "Time(hr)", "real SOC", caplegend, 1)
    #                 else:
    #                     dchgrealcap = " "
    #                     dchgmaxtime = " "
    #                 if self.saveok.isChecked():
    #                     cycoutputdata = pd.DataFrame(
    #                         {'name': namelist[-1],
    #                         'cycle': [str(i)],
    #                         'Chg_realSOC': [chgrealcap],
    #                         'Dchg_realSOC': [dchgrealcap],
    #                         'Chg time(min)': [chgmaxtime],
    #                         'Dchg time(min)': [dchgmaxtime]})
    #                     cycoutputdf = cycoutputdf._append(cycoutputdata)
    #                     # 충전 Profile 추출용
    #                     if hasattr(temp, "ChgProfile"):
    #                         chgoutputdf = chgoutputdf._append(temp.ChgProfile)
    #                     # 방전 Profile 추출용
    #                     if hasattr(temp, "DchgProfile"):
    #                         dchgoutputdf = dchgoutputdf._append(temp.DchgProfile)
    #         if self.saveok.isChecked() and save_file_name:
    #             cycoutputdf.to_excel(writer, sheet_name="cycle")
    #             chgoutputdf.to_excel(writer, sheet_name="chg")
    #             dchgoutputdf.to_excel(writer, sheet_name="dchg")
    #             writer.close()
    #         self.progressBar.setValue(100)
    #         fig.legend()
    #         plt.suptitle(namelist[-1], fontsize= 15, fontweight='bold')
    #         plt.tight_layout(pad=1, w_pad=1, h_pad=1)
    #         plt.subplots_adjust(right=0.8)
    #         plt.show()
    #         output_fig(self.figsaveok, namelist[-1])

    # def SetlogcycConfirmbutton(self):
    #     '''
    #     Act program Set log
    #     0:[TIME] 1: IMEI 2: Binary version 3: Capacity 4: cisd_fullcaprep_max 5: batt_charging_source
    #     6: charging_type 7: voltage_now 8: voltage_avg 9: current_now 10: current_avg
    #     11: battery_temp 12: ac_temp 13: temperature 14: battery_cycle 15: battery_charger_status
    #     16: batt_slate_mode 17: fg_asoc 18: fg_cycle 19: BIG 20: Little
    #     21: G3D 22: ISP 23: curr_5 24: wc_vrect 25: wc_vout
    #     26: dchg_temp 27: dchg_temp_adc 28: direct_charging_iin 29: AP CUR_CH0 30: AP CUR_CH1
    #     31: AP CUR_CH2 32: AP CUR_CH3 33: AP CUR_CH4 34: AP CUR_CH5 35: AP CUR_CH6
    #     36: AP CUR_CH7 37: AP POW_CH0 38: AP POW_CH1 39: AP POW_CH2 40: AP POW_CH3
    #     41: AP POW_CH4 42: AP POW_CH5 43: AP POW_CH6 44: AP POW_CH7 45: cisd_data
    #     46: LRP 47: USB_TEMP
    #     '''
    #     global writer
    #     root = Tk()
    #     root.withdraw()
    #     setxscale = int(self.setcyclexscale.text())
    #     self.SetlogcycConfirm.setDisabled(True)
    #     datafilepath = filedialog.askdirectory(initialdir="d://", title="Choose Test files")
    #     self.SetlogcycConfirm.setEnabled(True)
    #     if datafilepath:
    #         fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(4, 3))
    #         filecount = 0
    #         graphcolor = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    #         if self.saveok.isChecked():
    #             save_file_name = filedialog.asksaveasfilename(initialdir="D://", title="Save File Name", defaultextension=".xlsx")
    #             if save_file_name:
    #                 writer = pd.ExcelWriter(save_file_name, engine="xlsxwriter")
    #         rawdf = pd.DataFrame()
    #         df = pd.DataFrame()
    #         subfile = [f for f in os.listdir(datafilepath)]
    #         #폴더내의 파일 전체 취합
    #         for filename in subfile:
    #             rawdf = pd.read_csv(datafilepath + "/" + filename, on_bad_lines='skip')
    #             rawdf = rawdf.iloc[:-1]
    #             df = pd.concat([df, rawdf], axis=0, ignore_index=True)
    #             # progress bar 관련
    #             filecountmax = len(subfile)
    #             progressdata = filecount/filecountmax * 100
    #             filecount = filecount + 1
    #             self.progressBar.setValue(int(progressdata))
    #         if not df.empty:
    #             # 사이클을 위한 데이터 추출
    #             df.columns = df.columns.str.replace('[^A-Za-z0-9_]+', '', regex=True)
    #             df = df[["battery_cycle", "fg_cycle", "fg_asoc"]]
    #             # 중복 항목 제거 및 index reset
    #             df = df.drop_duplicates(subset='battery_cycle')
    #             df = df.sort_values(by="battery_cycle")
    #             df.reset_index()
    #             #그래프그리기
    #             graph_cycle(df["battery_cycle"], df["fg_asoc"], ax1, 80, 105, 5, "Cycle", "Discharge Capacity Ratio", "ASOC",
    #                         setxscale, graphcolor[0])
    #             if self.saveok.isChecked() and save_file_name:
    #                 df.to_excel(writer, sheet_name="SETcycle")
    #                 writer.close()
    #         self.progressBar.setValue(100)
    #         fig.legend()
    #         plt.tight_layout(pad=1, w_pad=1, h_pad=1)
    #         plt.show()

    def set_confirm_button(self):
        'Battery Status data log'
        '''0:Time 1:Level 2:Charging 3:Temperature(BA) 4:PlugType 5:Speed
        6:Voltage(mV) 7:Temperature(CHG) 8:Temperature(AP) 9:Temperature(Coil) 10:Ctype(Etc)-VOL
        11:Ctype(Etc)-ChargCur 12:Ctype(Etc)-Wire_Vout 13:Ctype(Etc)-Wire_Vrect 14:Temperature(CHG ADC) 15:Temperature(Coil ADC) 
        16:Temperature(BA ADC) 17:SafetyTimer 18:USB_Thermistor 19:SIOP_Level 20:Battery_Cycle
        21:Fg_Cycle 22:Charge_Time_Remaining 23:IIn 24:Temperature(DC) 25:Temperature(DC ADC) 
        26:DC Step 27:DC Status 28:Main Voltage 29:Sub Voltage 30:Main Current Now 
        31:Sub Current Now 32:Temperature(SUB Batt) 33:Temperature(SUB Batt ADC) 34:Current Avg. 35:ASOC1 
        36:Full Cap Nom 37:ASOC2 38:LRP 39:Raw SOC (%) 40:V avg (mV) 41:WC_Freq. 
        42:WC_Tx ID 43:Uno Vout 44:WC_Iin/Iout 45:Power 46:WC_Rx type 47:BSOH 48:Wireless 2.0 auth status 49:Full Voltage 50:Recharging Voltage 
        51:Full Cap Rep 52:CMD DATA 53:Temperature(AP ADC) 54:Battery Cycle Sub 55:charge status 56:Charging Cable 57:Fan Step 58:Fan Rpm
        59:Main Vchg 60:Sub Vchg 61:err_wthm
        '''
        global writer
        root = Tk()
        root.withdraw()
        self.SetConfirm.setDisabled(True)
        datafilepath = filedialog.askopenfilenames(initialdir="d://", title="Choose Test files")
        self.SetConfirm.setEnabled(True)
        if datafilepath:
            recentcycno = int(self.recentcycleno.text())
            filecount = 0
            if self.saveok.isChecked():
                save_file_name = filedialog.asksaveasfilename(initialdir="D://", title="Save File Name", defaultextension=".xlsx")
                if save_file_name:
                    writer = pd.ExcelWriter(save_file_name, engine="xlsxwriter")
                cycoutputdf = pd.DataFrame({'name': [''], 'cycle': [''], 'Chg_realSOC': [''], 'Dchg_realSOC': [''],
                                            'Chg time(min)': [''], 'Dchg time(min)': ['']})
                chgoutputdf = pd.DataFrame()
                dchgoutputdf = pd.DataFrame()
            for filepath in datafilepath:
                mincapa = int(self.SetMincapacity.text())
                fig, ax = plt.subplots(nrows=5, ncols=2, figsize=(18, 10))
                tab = QtWidgets.QWidget()
                tab_layout = QtWidgets.QVBoxLayout(tab)
                canvas = FigureCanvas(fig)
                toolbar = NavigationToolbar(canvas, None)
                chkcyc = set_act_ect_battery_status_cycle(filepath, self.realcyc.isChecked(), recentcycno,
                                                          self.allcycle.isChecked(), self.manualcycle.isChecked(), self.manualcycleno)
                filecountmax = len(datafilepath)
                filecount = filecount + 1
                # 전체 사이클과 최근 사이클 기준 설정
                if self.allcycle.isChecked() == True:
                    cyclecountmax = range(chkcyc[0], chkcyc[1] + 1)
                elif self.manualcycle.isChecked() == True:
                    manualcyclenochk = list(map(int, (self.manualcycleno.text().split())))
                    if len(manualcyclenochk) > 2:
                        manualcyclenochk = [x for x in manualcyclenochk if (x >= chkcyc[0] and x <= chkcyc[1])]
                        cyclecountmax = manualcyclenochk
                    else:
                        cycmin = max(chkcyc[0], manualcyclenochk[0])
                        cycmax = min(chkcyc[1], manualcyclenochk[1])
                        cyclecountmax = range(cycmin, cycmax + 1)
                else:
                # 최근 20 cycle 기준으로 설정
                    if (chkcyc[1] - chkcyc[0]) > recentcycno:
                        cyclecountmax = range(chkcyc[1] - recentcycno , chkcyc[1] + 1)
                    else:
                        cyclecountmax = range(chkcyc[0], chkcyc[1] + 1)
                namelist = filepath.split("/")[-1]
                for i in cyclecountmax:
                    temp = set_battery_status_log_Profile(chkcyc[2].Profile, mincapa, i, chkcyc[2].set)
                    # progressdata = ((filecount - 1) + (i - cyclecountmax[0] + 1)/len(cyclecountmax))/filecountmax * 100
                    progressdata = progress(filecount, filecountmax, (cyclecountmax[0] - i), len(cyclecountmax), 1, 1)
                    self.progressBar.setValue(int(progressdata))
                    if hasattr(temp, "ChgProfile"):
                        chgrealcap = str(round(temp.ChgProfile.SOC2.max(), 2))
                        chgmaxtime = str(round(temp.ChgProfile.Time.max() * 60, 1))
                        caplegend = "{:3}".format(str(i)) + " C:" + "{:8}".format(chgrealcap)
                        temp.ChgProfile = temp.ChgProfile[(temp.ChgProfile["Time"] < 4)]
                        graph_set_profile(temp.ChgProfile.Time * 60, temp.ChgProfile.Vol, ax[0, 0], 3.6, 4.8, 0.2, "Time(hr)", "Voltage (V)", "", 0, 0, 120, 10)
                        graph_set_profile(temp.ChgProfile.Time * 60, temp.ChgProfile.Curr/1000, ax[1, 0], 0, 3.2, 0.2, "Time(hr)", "Curr", "", 0, 0, 120, 10)
                        graph_set_profile(temp.ChgProfile.Time * 60, temp.ChgProfile.Temp, ax[2, 0], 20, 50, 4, "Time(hr)", "temp.", "", 0, 0, 120, 10)
                        graph_set_profile(temp.ChgProfile.Time * 60, temp.ChgProfile.SOC, ax[3, 0], 0, 120, 10, "Time(hr)", "SOC", "", 0, 0, 120, 10)
                        graph_set_profile(temp.ChgProfile.Time * 60, temp.ChgProfile.SOC2/1000, ax[4, 0], 0, 120, 10, "Time(hr)", "real SOC", "", 0, 0, 120, 10)
                    else:
                        chgrealcap = " "
                        chgmaxtime = " "
                    if hasattr(temp, "DchgProfile"):
                        dchgrealcap = str(round(temp.DchgProfile.SOC2.max(), 2))
                        dchgmaxtime = str(round(temp.DchgProfile.Time.max() * 60, 1))
                        caplegend = caplegend + " D:" + "{:6}".format(dchgrealcap)
                        graph_set_profile(temp.DchgProfile.Time, temp.DchgProfile.Vol, ax[0, 1], 3.0, 4.6, 0.2, "Time(hr)", "Voltage (V)", "", 0, 0, 11, 1)
                        graph_set_profile(temp.DchgProfile.Time, temp.DchgProfile.Curr/1000, ax[1, 1], -0.6, 0.1, 0.1, "Time(hr)", "Curr", "", 0, 0, 11, 1)
                        graph_set_profile(temp.DchgProfile.Time, temp.DchgProfile.Temp, ax[2, 1], 20, 50, 4, "Time(hr)", "temp.", "", 0, 0, 11, 1)
                        graph_set_profile(temp.DchgProfile.Time, temp.DchgProfile.SOC, ax[3, 1], 0, 120, 10, "Time(hr)", "SOC", "", 0, 0, 11, 1)
                        if self.saveok.isChecked():
                            graph_set_profile(temp.DchgProfile.Time, 100 - temp.DchgProfile.SOC2/1000, ax[4, 1], 0, 120, 10, "Time(hr)", "real SOC",
                                              i, 0, 0, 11, 1)
                        else:
                            graph_set_profile(temp.DchgProfile.Time, 100 - temp.DchgProfile.SOC2/1000, ax[4, 1], 0, 120, 10, "Time(hr)", "real SOC",
                                              caplegend, 0, 0, 11, 1)
                    else:
                        dchgrealcap = " "
                        dchgmaxtime = " "
                    if self.saveok.isChecked():
                        cycoutputdata = pd.DataFrame(
                            {'name': namelist,
                            'cycle': [str(i)],
                            'Chg_realSOC': [chgrealcap],
                            'Dchg_realSOC': [dchgrealcap],
                            'Chg time(min)': [chgmaxtime],
                            'Dchg time(min)': [dchgmaxtime]})
                        cycoutputdf = cycoutputdf._append(cycoutputdata)
                        # 충전 Profile 추출용
                        if hasattr(temp, "ChgProfile"):
                            chgoutputdf = chgoutputdf._append(temp.ChgProfile)
                        # 방전 Profile 추출용
                        if hasattr(temp, "DchgProfile"):
                            dchgoutputdf = dchgoutputdf._append(temp.DchgProfile)
                    # Chgnamelist = datafilepath.split("/")
                    for i in range(2):
                        for j in range(4):
                            # X축 레이블 제거
                            ax[j, i].set_xlabel('')
                            # X축 틱 레이블 제거
                            ax[j, i].set_xticklabels([])
                    tab_layout.addWidget(toolbar)
                    tab_layout.addWidget(canvas)
                    self.set_tab.addTab(tab, namelist)
                    self.set_tab.setCurrentWidget(tab)
                    plt.tight_layout(pad=1, w_pad=1, h_pad=1)
                if self.saveok.isChecked() and save_file_name:
                    cycoutputdf.to_excel(writer, sheet_name="cycle")
                    chgoutputdf.to_excel(writer, sheet_name="chg")
                    dchgoutputdf.to_excel(writer, sheet_name="dchg")
                    writer.close()
                fig.legend()
                plt.subplots_adjust(right=0.8)
                # plt.suptitle(Chgnamelist[-1], fontsize= 15, fontweight='bold')
                plt.tight_layout(pad=1, w_pad=1, h_pad=1)
                output_fig(self.figsaveok, namelist)
                plt.close()
            self.progressBar.setValue(100)

    def set_cycle_button(self):
        'Battery Status data log'
        '''0:Time 1:Level 2:Charging 3:Temperature(BA) 4:PlugType 5:Speed
        6:Voltage(mV) 7:Temperature(CHG) 8:Temperature(AP) 9:Temperature(Coil) 10:Ctype(Etc)-VOL
        11:Ctype(Etc)-ChargCur 12:Ctype(Etc)-Wire_Vout 13:Ctype(Etc)-Wire_Vrect 14:Temperature(CHG ADC) 15:Temperature(Coil ADC) 
        16:Temperature(BA ADC) 17:SafetyTimer 18:USB_Thermistor 19:SIOP_Level 20:Battery_Cycle
        21:Fg_Cycle 22:Charge_Time_Remaining 23:IIn 24:Temperature(DC) 25:Temperature(DC ADC) 
        26:DC Step 27:DC Status 28:Main Voltage 29:Sub Voltage 30:Main Current Now 
        31:Sub Current Now 32:Temperature(SUB Batt) 33:Temperature(SUB Batt ADC) 34:Current Avg. 35:ASOC1 
        36:Full Cap Nom 37:ASOC2 38:LRP 39:Raw SOC (%) 40:V avg (mV) 41:WC_Freq. 
        42:WC_Tx ID 43:Uno Vout 44:WC_Iin/Iout 45:Power 46:WC_Rx type 47:BSOH 48:Wireless 2.0 auth status 49:Full Voltage 50:Recharging Voltage 
        51:Full Cap Rep 52:CMD DATA 53:Temperature(AP ADC) 54:Battery Cycle Sub 55:charge status 56:Charging Cable 57:Fan Step 58:Fan Rpm
        59:Main Vchg 60:Sub Vchg 61:err_wthm
        '''
        global writer
        root = Tk()
        root.withdraw()
        setxscale = int(self.setcyclexscale.text())
        self.SetCycle.setDisabled(True)
        datafilepath = filedialog.askdirectory(initialdir="d://", title="Choose Test files")
        self.SetCycle.setEnabled(True)
        if datafilepath:
            fig, ((ax1), (ax2)) = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))
            filecount = 0
            mincapa = int(self.SetMincapacity.text())
            graphcolor = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
            if self.saveok.isChecked():
                save_file_name = filedialog.asksaveasfilename(initialdir="D://", title="Save File Name", defaultextension=".xlsx")
                if save_file_name:
                    writer = pd.ExcelWriter(save_file_name, engine="xlsxwriter")
            rawdf = pd.DataFrame()
            df = pd.DataFrame()
            subfile = [f for f in os.listdir(datafilepath)]
            #폴더내의 파일 전체 취합
            for filename in subfile:
                rawdf = pd.read_csv(datafilepath + "/" + filename, on_bad_lines='skip')
                rawdf = rawdf.iloc[:-1]
                df = pd.concat([df, rawdf], axis=0, ignore_index=True)
                # progress bar 관련
                filecountmax = len(subfile)
                progressdata = filecount/filecountmax * 100
                filecount = filecount + 1
                self.progressBar.setValue(int(progressdata))
            if not df.empty:
                # 사이클을 위한 데이터 추출
                df = df[["Battery_Cycle", "Fg_Cycle", "ASOC1", "Full Cap Nom", "ASOC2", "BSOH", "Full Cap Rep", "Battery Cycle Sub"]]
                # 중복 항목 제거 및 index reset
                df = df.drop_duplicates(subset='Battery_Cycle')
                df = df.sort_values(by="Battery_Cycle")
                df["ASOC3"] = df["Full Cap Rep"]/mincapa*100
                df.reset_index()
                #그래프그리기
                graph_cycle(df["Battery_Cycle"], df["ASOC1"], ax1, 80, 105, 5, "Cycle", "Discharge Capacity Ratio", "ASOC1",
                            setxscale, graphcolor[0])
                graph_cycle(df["Battery_Cycle"], df["BSOH"], ax1, 80, 105, 5, "Cycle", "Discharge Capacity Ratio", "BSOH",
                            setxscale, graphcolor[1])
                graph_cycle(df["Battery_Cycle"], df["ASOC3"], ax1, 80, 105, 5, "Cycle", "Discharge Capacity Ratio", "ASOC3",
                            setxscale, graphcolor[4])
                graph_cycle(df["Battery_Cycle"], df["Full Cap Nom"], ax2, 4000, 5000, 50, "Cycle", "Capacity(mAh)", "Full Cap Nom",
                            setxscale, graphcolor[2])
                graph_cycle(df["Battery_Cycle"], df["Full Cap Rep"], ax2, 4000, 5000, 50, "Cycle", "Capacity(mAh)", "Full Cap Rep",
                            setxscale, graphcolor[3])
                if self.saveok.isChecked() and save_file_name:
                    df.to_excel(writer, sheet_name="SETcycle")
                    writer.close()
            self.progressBar.setValue(100)
            fig.legend()
            plt.tight_layout(pad=1, w_pad=1, h_pad=1)
            plt.show()

    def ect_data(self, datafilepath, state):
        # 260106 App log
        # 0:Time,  1:voltage_now(mV),  2:V avg (mV),  3:Ctype(Etc)-ChargCur,  4:Current Avg.,  5:comp_current_avg,  6:current offset,
        # 7:Level,  8:fg_full_voltage,  9:cc_info,  10:Charging,  11:Battery_Cycle,  12:sleepTime,  13:diffTime,  14:Temperature(BA),
        # 15:temp_ambient,  16:batt_temp,  17:batt_temp_adc,  18:batt_temp_raw,  19:batt_wpc_temp,  20:batt_wpc_temp_adc,  21:chg_temp,
        # 22:chg_temp_adc,  23:dchg_temp,  24:dchg_temp_adc,  25:usb_temp,  26:usb_temp_adc,  27:cutOff(mA),  28:Avg(mA),  29:Queue[0],
        # 30:Queue[1],  31:Queue[2],  32:Queue[3],  33:Queue[4],  34:Queue[5],  35:Queue[6],  36:Queue[7],  37:Queue[8],  38:Queue[9],
        # 39:CNT,  40:ectSOC,  41:RSOC,  42:SOC_RE,  43:SOC_EDV,  44:RSOH,  45:SOH,  46:AnodePotential,  47:SOH_dR,  48:SOH_CA,  49:SOH_X,
        # 50:SC_VALUE,  51:SC_SCORE,  52:SC_Grade,  53:SC_V_Acc,  54:SC_V_Avg,  55:avg_I_ISC,  56:avg_R_ISC,  57:avg_R_ISC_min,  58:LUT_VOLT0,
        # 59:LUT_VOLT1,  60:LUT_VOLT2,  61:LUT_VOLT3,  62:T_move,  63:OCV, 
        Profile = pd.read_csv(datafilepath, sep=",", on_bad_lines='skip', skiprows = 1,  encoding="UTF-8")
        if Profile.iloc[0,0] == "Time":
            Profile = pd.read_csv(datafilepath, sep=",", skiprows = 1, on_bad_lines='skip')
        Profile.columns = Profile.columns.str.replace('[^A-Za-z0-9_]+', '', regex=True)
        Profile = Profile[['Time', 'voltage_nowmV', 'CtypeEtcChargCur', 'CurrentAvg', 'TemperatureBA', 'Level', 'ectSOC',
                           'RSOC', 'SOC_RE', 'Charging', 'Battery_Cycle', 'AnodePotential', 'SC_SCORE', 'VavgmV', 'LUT_VOLT0',
                           'LUT_VOLT1', 'LUT_VOLT2', 'LUT_VOLT3']]
        Profile.columns = ['Time', 'Vol', 'Curr', 'CurrAvg', 'Temp', 'SOC', 'SOCectraw',
                        'RSOCect', 'SOCect', 'Type', 'Cyc', 'anodeE', 'short', 'Vavg', '1stepV', '2stepV',
                        '3stepV', '4stepV']
        Profile.Time = '20'+ Profile['Time'].astype(str)
        Profile = Profile[:-1]
        cycmin = int(Profile.Cyc.min())
        cycmax = int(Profile.Cyc.max())
        if self.realcyc.isChecked() == 0 and state == "profile":
            if cycmin != cycmax:
                # 'Unplugged' 또는 ' NONE'에서 'AC' 또는 ' PDIC_APDO'로 전환되는 지점을 찾습니다.
                plug_change = (
                    (Profile['Type'].shift(1).isin([" Discharging"])) &
                    (Profile['Type'].isin([" Charging", " Full"]))
                )
                # 'Battery_Cycle'의 초기값을 설정합니다.
                Profile['Cyc'] = cycmin
                # 전환 지점에서 1씩 증가하는 값을 누적합으로 계산하여 적용합니다.
                # cumsum() 함수는 True(1)인 값의 누적합을 효율적으로 계산합니다.
                Profile['Cyc'] += plug_change.cumsum()
        cycmax = int(Profile.Cyc.max())
        if not Profile.empty:
            # 시간 확인
            Profile["Time"] = pd.to_datetime(Profile["Time"], format="%Y%m%d %H:%M:%S.%f")
            Profile["Time"] = Profile["Time"] - Profile["Time"].loc[0]
            Profile["Time"] = Profile["Time"].dt.total_seconds().div(3600).astype(float)
            Profile['Curr']=Profile['Curr'].apply(float)/1000
            Profile['CurrAvg']=Profile['CurrAvg'].apply(float)/1000
            Profile['SOC']=Profile['SOC'].apply(float)
            Profile['Vol']=Profile['Vol'].apply(float)/1000
            Profile['Temp']=Profile['Temp'].apply(float)
            Profile['Cyc']=Profile['Cyc'].apply(int)
            Profile['anodeE']=Profile['anodeE'].apply(float)/1000
            Profile['SOCectraw']=Profile['SOCectraw'].apply(float)/10
            Profile['RSOCect']=Profile['RSOCect'].apply(float)/10
            Profile['SOCect']=Profile['SOCect'].apply(float)/10
            Profile["delTime"] = 0
            Profile["delCap"] = 0
            Profile["SOCref"] = 0
            Profile["delCapAvg"] = 0
            Profile["SOCrefAvg"] = 0
        return Profile
    
    def ect_short_button(self):
        global writer
        root = Tk()
        root.withdraw()
        self.ECTShort.setDisabled(True)
        datafilepaths = filedialog.askopenfilenames(initialdir="d://", title="Choose Test files")
        self.ECTShort.setEnabled(True)
        if datafilepaths:
            for datafilepath in datafilepaths:
            # 최근 사이클 산정 및 전체 사이클 적용여부 확인
                mincapa = int(self.SetMincapacity.text())
                if self.saveok.isChecked():
                    save_file_name = filedialog.asksaveasfilename(initialdir="D://", title="Save File Name", defaultextension=".xlsx")
                    if save_file_name:
                        writer = pd.ExcelWriter(save_file_name, engine="xlsxwriter")
                fig, ax = plt.subplots(nrows=5, ncols=1, figsize=(6, 10))
                tab = QtWidgets.QWidget()
                tab_layout = QtWidgets.QVBoxLayout(tab)
                canvas = FigureCanvas(fig)
                toolbar = NavigationToolbar(canvas, None)
                Profile = self.ect_data(datafilepath, "short")
            #Short Profile 확인용
                graph_set_profile(Profile.Time, Profile.Vol, ax[0], 3.0, 4.8, 0.2, "Time(hr)", "Voltage (V)", "", 0, 0, 0, 0)
                # graph_set(Profile.Time, Profile.anodeE, ax2, -0.1, 0.8, 0.1, "Time(hr)", "anodeE", "", 99)
                graph_set_profile(Profile.Time, Profile.CurrAvg, ax[1], -10, 11, 2, "Time(hr)", "Curr(A)", "", 0, 0, 0, 0)
                graph_set_profile(Profile.Time, Profile.Temp, ax[2], 20, 50, 4, "Time(hr)", "temp.(℃)", "", 0, 0, 0, 0)
                graph_set_profile(Profile.Time, Profile.SOC, ax[3], 0, 120, 10, "Time(hr)", "SOC/SOCect", "", 0, 0, 0, 0)
                graph_set_profile(Profile.Time, Profile.SOCect, ax[3], 0, 120, 10, "Time(hr)", "SOC/SOCect", "", 1, 0, 0, 0)
                # graph_set_profile(Profile.Time, Profile.SOCectraw, ax[3], 0, 120, 10, "Time(hr)", "SOC/SOCect/SOCectraw", "", 2, 0, 0, 0)
            # Short 관련
                graph_set_profile(Profile.Time, Profile.short, ax[4], 0, 6, 1, "Time(hr)", "Short Score", "", 0, 0, 0, 0)
                # 마지막 행을 제외한 각 서브플롯 설정
                for i in range(4):
                    # X축 레이블 제거
                    ax[i].set_xlabel('')
                    # X축 틱 레이블 제거
                    ax[i].set_xticklabels([])
                Chgnamelist = datafilepath.split("/")
                tab_layout.addWidget(toolbar)
                tab_layout.addWidget(canvas)
                self.set_tab.addTab(tab, Chgnamelist[-1])
                self.set_tab.setCurrentWidget(tab)
                plt.tight_layout(pad=1, w_pad=1, h_pad=1)
            if self.saveok.isChecked() and save_file_name:
                Profile.to_excel(writer)
                writer.close()
            fig.legend()
            plt.subplots_adjust(right=0.8)
            # plt.suptitle(Chgnamelist[-1], fontsize= 15, fontweight='bold')
            plt.tight_layout(pad=1, w_pad=1, h_pad=1)
            output_fig(self.figsaveok,Chgnamelist)
            plt.close()
        self.progressBar.setValue(100)

    def ect_soc_button(self):
        global writer
        root = Tk()
        root.withdraw()
        self.ECTSOC.setDisabled(True)
        datafilepaths = filedialog.askopenfilenames(initialdir="d://", title="Choose Test files")
        self.ECTSOC.setEnabled(True)
        filecount = 0
        filecountmax = len(datafilepaths)
        if datafilepaths:
            for datafilepath in datafilepaths:
            # 최근 사이클 산정 및 전체 사이클 적용여부 확인
                recentcycno = int(self.recentcycleno.text())
                # mincapa = int(self.SetMincapacity.text())
                setoffvol = self.setoffvoltage.text()
                if self.saveok.isChecked():
                    save_file_name = filedialog.asksaveasfilename(initialdir="D://", title="Save File Name", defaultextension=".xlsx")
                    if save_file_name:
                        writer = pd.ExcelWriter(save_file_name, engine="xlsxwriter")
                    dfdchg = pd.DataFrame()
                fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(18, 10))
                tab = QtWidgets.QWidget()
                tab_layout = QtWidgets.QVBoxLayout(tab)
                canvas = FigureCanvas(fig)
                toolbar = NavigationToolbar(canvas, None)
                Profile = self.ect_data(datafilepath, "profile")
                Profile["Vavg_est"] = Profile["Vol"].rolling(window=45, min_periods=1).mean()
            # 전체 사이클과 최근 사이클 기준 설정
                if not Profile.empty:
                    # 시간 확인
                    if setoffvol != "" and (Profile["Vavg_est"].min() < float(setoffvol)) :
                        cutoff_index = Profile.index[Profile["Vavg_est"] <= float(setoffvol)].tolist()
                        Profile = Profile.loc[:cutoff_index[0]]
                    tempDchgProfile = Profile[Profile.Type == " Discharging"]
            # 전체 사이클과 최근 사이클 기준 설정
                if self.allcycle.isChecked() == True:
                    cyclecountmax = range(Profile.Cyc.min(), Profile.Cyc.max()+1)
                elif self.manualcycle.isChecked() == True:
                    manualcyclenochk = list(map(int, (self.manualcycleno.text().split())))
                    if len(manualcyclenochk) > 2:
                        manualcyclenochk = [x for x in manualcyclenochk if (x >= Profile.Cyc.min() and x <= Profile.Cyc.max())]
                        cyclecountmax = manualcyclenochk
                    else:
                        cycmin = max(Profile.Cyc.min(), manualcyclenochk[0])
                        cycmax = min(Profile.Cyc.max(), manualcyclenochk[1])
                        cyclecountmax = range(cycmin, cycmax + 1)
                else: # 최근 20 cycle 기준으로 설정
                    if (Profile.Cyc.max() - Profile.Cyc.min()) > recentcycno:
                        cyclecountmax = range(Profile.Cyc.max() - recentcycno , Profile.Cyc.max() + 1)
                    elif Profile.Cyc.max() == Profile.Cyc.min():
                        cyclecountmax = [Profile.Cyc.min()]
                    else:
                        cyclecountmax = range(Profile.Cyc.min(), Profile.Cyc.max() + 1)
                for i in cyclecountmax:
                    DchgProfile = tempDchgProfile[(tempDchgProfile.Cyc == i)]
                    DchgProfile = DchgProfile.reset_index()
                    DchgProfile.delTime = DchgProfile.Time.diff()
                    DchgProfile.delCap = DchgProfile.delTime * DchgProfile.Curr
                    DchgProfile.delCapAvg = DchgProfile.delTime * DchgProfile.CurrAvg
                    DchgRealCap = abs(DchgProfile.delCap.cumsum())
                    DchgRealAvgCap = abs(DchgProfile.delCapAvg.cumsum())
                    DchgSOCrefmax = DchgRealCap / DchgRealCap.max()
                    DchgSOCrefAvgmax = DchgRealAvgCap / DchgRealAvgCap.max()
                    DchgProfile.SOCref = 100 - DchgSOCrefmax * 100
                    DchgProfile.SOCrefAvg = 100 - DchgSOCrefAvgmax * 100
                    if DchgRealAvgCap.max() > 1:
                        self.socmaxcapacity.setText(str(int(DchgRealAvgCap.max() * 1000)))
                    if not DchgProfile.empty:
                        DchgProfile.Time = DchgProfile.Time - DchgProfile.Time.loc[0]
                    DchgProfile.SOCError = DchgProfile.SOCrefAvg - DchgProfile.SOC
                    DchgProfile.SOCectError = DchgProfile.SOCrefAvg - DchgProfile.SOCect
                    progressdata = progress(filecount, filecountmax, (cyclecountmax[0] - i), len(cyclecountmax), 1, 1)
                    self.progressBar.setValue(int(progressdata))
                    self.socerrormax.setText(str(round(DchgProfile.SOCError.abs().max(),3)))
                    self.socerroravg.setText(str(round(DchgProfile.SOCError.abs().mean(),3)))
                    self.ectsocerrormax.setText(str(round(DchgProfile.SOCectError.abs().max(),3)))
                    self.ectsocerroravg.setText(str(round(DchgProfile.SOCectError.abs().mean(),3)))
                # 방전 관련
                    graph_soc_set(DchgProfile.Time, DchgProfile.Vol, ax[0, 0], 3.0, 4.8, 0.2, "Time(hr)", "Voltage (V)", "", 1)
                    graph_soc_set(DchgProfile.Time, DchgProfile.Vavg_est, ax[0, 0], 3.0, 4.8, 0.2, "Time(hr)", "Voltage (V)", "", 3)
                    graph_soc_set(DchgProfile.Time, DchgProfile.anodeE, ax[1, 0], 0, 0.8, 0.1, "Time(hr)", "Anode Voltage (V)", "", 1)
                    graph_soc_set(DchgProfile.Time, DchgProfile.CurrAvg, ax[2, 0], 0, 0, 0, "Time(hr)", "Current(A)", "", 1)
                    graph_soc_set(DchgProfile.Time, DchgProfile.Temp, ax[3, 0], -20, 60, 10, "Time(hr)", "Temperature (℃)", "", 1)
                # SOC 비교
                    graph_soc_set(DchgProfile.Time, DchgProfile.SOCectraw, ax[0, 1], 0, 110, 10, "Time(hr)", "ASOC_ect", "ASOC_ect", 6)
                    graph_soc_set(DchgProfile.Time, DchgProfile.RSOCect, ax[0, 1], 0, 110, 10, "Time(hr)", "SOCraw_ect", "SOCraw_ect", 7)
                    graph_soc_set(DchgProfile.Time, DchgProfile.SOCect, ax[0, 1], 0, 110, 10, "Time(hr)", "SOC_ect", "SOC_ect", 4)
                    graph_soc_set(DchgProfile.Time, DchgProfile.SOC, ax[1, 1], 0, 110, 10, "Time(hr)", "SOC/ SOC_ect/ SOC_ref", "SOC", 3)
                    graph_soc_set(DchgProfile.Time, DchgProfile.SOCect, ax[1, 1], 0, 110, 10, "Time(hr)", "SOC/ SOC_ect/ SOC_ref", "SOC_ect", 4)
                    graph_soc_set(DchgProfile.Time, DchgProfile.SOCrefAvg, ax[1, 1], 0, 110, 10, "Time(hr)", "SOC/SOCect/SOCref", "SOC_ref", 5)
                    graph_soc_set(DchgProfile.Time, DchgProfile.SOCError, ax[2, 1], -10, 11, 2, "Time(hr)", "Error(%)", "SOC", 3)
                    graph_soc_set(DchgProfile.Time, DchgProfile.SOCectError, ax[2, 1], -10, 11, 2, "Time(hr)", "Error(%)", "SOC_ect", 4)
                    graph_soc_err(DchgProfile.SOCrefAvg, DchgProfile.SOCError, ax[3, 1], -10, 11, 2, "SOCref", "Error(%)", "SOC", 3)
                    graph_soc_err(DchgProfile.SOCrefAvg, DchgProfile.SOCectError, ax[3, 1], -10, 11, 2, "SOCref", "Error(%)", "SOC_ect", 4)
                if self.saveok.isChecked() and save_file_name:
                    dfdchg = dfdchg._append(DchgProfile)
                if self.saveok.isChecked() and save_file_name:
                    dfdchg.to_excel(writer, sheet_name="dchg")
                    writer.close()
                tab_name_list = datafilepath.split("/")[-1].split(".")[-2]
                ax[0, 0].legend(loc="lower left")
                ax[1, 0].legend(loc="upper left")
                ax[2, 0].legend(loc="lower right")
                ax[3, 0].legend(loc="upper right")
                ax[0, 1].legend(loc="upper right")
                ax[1, 1].legend(loc="upper right")
                ax[2, 1].legend(loc="upper left")
                ax[3, 1].legend(loc="upper left")
                tab_layout.addWidget(toolbar)
                tab_layout.addWidget(canvas)
                self.set_tab.addTab(tab, tab_name_list) 
                self.set_tab.setCurrentWidget(tab)
                plt.tight_layout(pad=1, w_pad=1, h_pad=1)
                output_fig(self.figsaveok, tab_name_list)
                plt.close()
                filecount = filecount + 1
            self.progressBar.setValue(100)

    def ect_set_profile_button(self):
        global writer
        root = Tk()
        root.withdraw()
        self.ECTSetProfile.setDisabled(True)
        datafilepaths = filedialog.askopenfilenames(initialdir="d://", title="Choose Test files")
        self.ECTSetProfile.setEnabled(True)
        if datafilepaths:
            for datafilepath in datafilepaths:
            # 최근 사이클 산정 및 전체 사이클 적용여부 확인
                recentcycno = int(self.recentcycleno.text())
                if self.saveok.isChecked():
                    save_file_name = filedialog.asksaveasfilename(initialdir="D://", title="Save File Name", defaultextension=".xlsx")
                    if save_file_name:
                        writer = pd.ExcelWriter(save_file_name, engine="xlsxwriter")
                    dfchg = pd.DataFrame()
                    dfdchg = pd.DataFrame()
                fig, ax = plt.subplots(nrows=5, ncols=2, figsize=(18, 10))
                # fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8), (ax9, ax10)) = plt.subplots(nrows=5, ncols=2, figsize=(18, 10))
                tab = QtWidgets.QWidget()
                tab_layout = QtWidgets.QVBoxLayout(tab)
                canvas = FigureCanvas(fig)
                toolbar = NavigationToolbar(canvas, None)
                Profile = self.ect_data(datafilepath, "profile")
            # 전체 사이클과 최근 사이클 기준 설정
                if not Profile.empty:
                    tempDchgProfile = Profile[Profile.Type == " Discharging"]
                    tempChgProfile = Profile[Profile.Type != " Discharging"]
            # 전체 사이클과 최근 사이클 기준 설정
                if self.allcycle.isChecked() == True:
                    cyclecountmax = range(Profile.Cyc.min(), Profile.Cyc.max()+1)
                elif self.manualcycle.isChecked() == True:
                    manualcyclenochk = list(map(int, (self.manualcycleno.text().split())))
                    if len(manualcyclenochk) > 2:
                        manualcyclenochk = [x for x in manualcyclenochk if (x >= Profile.Cyc.min() and x <= Profile.Cyc.max())]
                        cyclecountmax = manualcyclenochk
                    else:
                        cycmin = max(Profile.Cyc.min(), manualcyclenochk[0])
                        cycmax = min(Profile.Cyc.max(), manualcyclenochk[1])
                        cyclecountmax = range(cycmin, cycmax + 1)
                else:
            # 최근 20 cycle 기준으로 설정
                    if (Profile.Cyc.max() - Profile.Cyc.min()) > recentcycno:
                        cyclecountmax = range(Profile.Cyc.max() - recentcycno , Profile.Cyc.max() + 1)
                    elif Profile.Cyc.max() == Profile.Cyc.min():
                        cyclecountmax = [Profile.Cyc.min()]
                    else:
                        cyclecountmax = range(Profile.Cyc.min(), Profile.Cyc.max() + 1)
                for cycno in cyclecountmax:
                    DchgProfile = tempDchgProfile[(tempDchgProfile.Cyc == cycno)]
                    ChgProfile = tempChgProfile[(tempChgProfile.Cyc == cycno)]
                    DchgProfile = DchgProfile.reset_index()
                    ChgProfile = ChgProfile.reset_index()
                    DchgProfile.delTime = DchgProfile.Time.diff()
                    DchgProfile.delCap = DchgProfile.delTime * DchgProfile.Curr
                    SOCrefmax = abs(DchgProfile.delCap.cumsum() * 100)
                    DchgProfile.SOCref = 100 - SOCrefmax
                    DchgProfile.delCapAvg = DchgProfile.delTime * DchgProfile.CurrAvg
                    SOCrefAvgmax = abs(DchgProfile.delCapAvg.cumsum() * 100)
                    DchgProfile.SOCrefAvg = 100 - SOCrefAvgmax
                    ChgProfile.delTime = ChgProfile.Time.diff()
                    ChgProfile.delCap = ChgProfile.delTime * ChgProfile.Curr
                    ChgProfile.SOCref = 100-abs(ChgProfile.delCap.cumsum() * 100)
                    ChgProfile.delCapAvg = ChgProfile.delTime * ChgProfile.CurrAvg
                    ChgProfile.SOCrefAvg = 100-abs(ChgProfile.delCapAvg.cumsum() * 100)
                    if not DchgProfile.empty:
                        DchgProfile.Time = DchgProfile.Time - DchgProfile.Time.loc[0]
                    if not ChgProfile.empty:
                        ChgProfile.Time = ChgProfile.Time - ChgProfile.Time.loc[0]
                    progressdata = (cycno - cyclecountmax[0] + 1)/len(cyclecountmax) * 100
                    self.progressBar.setValue(int(progressdata))
                    ChgProfile = ChgProfile[(ChgProfile["Time"] < 4)]
                    DchgProfile.SOCError = DchgProfile.SOCrefAvg - DchgProfile.SOC
                    DchgProfile.SOCectError = DchgProfile.SOCrefAvg - DchgProfile.SOCect
                # 충전 관련
                    graph_set_profile(ChgProfile.Time, ChgProfile.Vol, ax[0, 0], 3.0, 4.8, 0.2, "Time(hr)", "Voltage (V)", "", 0, 0, 4, 1)
                    graph_set_profile(ChgProfile.Time, ChgProfile.anodeE, ax[1, 0], -0.1, 0.3, 0.05, "Time(hr)", "anodeE (V)", "", 0, 0, 4, 1)
                    graph_set_profile(ChgProfile.Time, ChgProfile.CurrAvg/1000, ax[2, 0], 0, 0, 0, "Time(hr)", "CurrAvg (A)", "", 0, 0, 4, 1)
                    graph_set_profile(ChgProfile.Time, ChgProfile.Temp, ax[3, 0], 20, 50, 4, "Time(hr)", "temp.(℃ )", "", 0, 0, 4, 1)
                    graph_set_profile(ChgProfile.Time, ChgProfile.SOC, ax[4, 0], 0, 120, 10, "Time(hr)", "SOC", "", 1, 0, 4, 1)
                    graph_set_profile(ChgProfile.Time, ChgProfile.SOCect, ax[4, 0], 0, 120, 10, "Time(hr)", "SOC", "", 2, 0, 4, 1)
                # 방전 관련
                    graph_set_profile(DchgProfile.Time, DchgProfile.Vol, ax[0, 1], 3.0, 4.8, 0.2, "Time(hr)", "Voltage (V)", "", 0, 0, 11, 1)
                    graph_set_profile(DchgProfile.Time, DchgProfile.anodeE, ax[1, 1], 0, 0.8, 0.1, "Time(hr)", "anodeE (V)", "", 0, 0, 11, 1)
                    graph_set_profile(DchgProfile.Time, DchgProfile.CurrAvg/1000, ax[2, 1], 0, 0, 0, "Time(hr)", "CurrAvg (A)", "", 0, 0, 11, 1)
                    graph_set_profile(DchgProfile.Time, DchgProfile.Temp, ax[3, 1], 20, 50, 4, "Time(hr)", "temp.(℃)", "", 0, 0, 11, 1)
                    graph_set_profile(DchgProfile.Time, DchgProfile.SOC, ax[4, 1], 0, 120, 10, "Time(hr)", "SOC", "", 1, 0, 11, 1)
                    graph_set_profile(DchgProfile.Time, DchgProfile.SOCect, ax[4, 1], 0, 120, 10, "Time(hr)", "SOC", "", 2, 0, 11, 1)
                # SOC 비교
                    if "1stepV" in ChgProfile.columns:
                        graph_set_guide(ChgProfile.Time, ChgProfile["1stepV"], ax[0 ,0], 3.0, 4.8, 0.2, "Time(hr)", "Voltage (V)", "", 0, 4, 1)
                        graph_set_guide(ChgProfile.Time, ChgProfile["2stepV"], ax[0, 0], 3.0, 4.8, 0.2, "Time(hr)", "Voltage (V)", "", 0, 4, 1)
                        graph_set_guide(ChgProfile.Time, ChgProfile["3stepV"], ax[0, 0], 3.0, 4.8, 0.2, "Time(hr)", "Voltage (V)", "", 0, 4, 1)
                        graph_set_guide(ChgProfile.Time, ChgProfile["4stepV"], ax[0, 0], 3.0, 4.8, 0.2, "Time(hr)", "Voltage (V)", "", 0, 4, 1)
                    plt.tight_layout(pad=1, w_pad=1, h_pad=1)
                for i in range(4):
                    for j in range(2):
                        # X축 레이블 제거
                        ax[i, j].set_xlabel('')
                        # X축 틱 레이블 제거
                        ax[i, j].set_xticklabels([])
                if self.saveok.isChecked() and save_file_name:
                    dfdchg = dfdchg._append(DchgProfile)
                    dfchg = dfchg._append(ChgProfile)
                if self.saveok.isChecked() and save_file_name:
                    if not self.chk_setcyc_sep.isChecked():
                        dfdchg.to_excel(writer, sheet_name="dchg")
                        dfchg.to_excel(writer, sheet_name="chg")
                    else:
                        Profile.to_excel(writer)
                    writer.close()
                fig.legend()
                plt.subplots_adjust(right=0.8)
                tab_name_list = datafilepath.split("/")[-1].split(".")[-2]
                # plt.suptitle(Chgnamelist[-1], fontsize=15)
                tab_layout.addWidget(toolbar)
                tab_layout.addWidget(canvas)
                self.set_tab.addTab(tab, tab_name_list) 
                self.set_tab.setCurrentWidget(tab)
                plt.tight_layout(pad=1, w_pad=1, h_pad=1)
                output_fig(self.figsaveok, tab_name_list)
                plt.close()
            self.progressBar.setValue(100)

    def ect_set_cycle_button(self):
        self.ECTSetCycle.setDisabled(True)
        global writer
        setxscale = int(self.setcyclexscale.text())
        subfile = []
        root = Tk()
        root.withdraw()
        datafilepath = filedialog.askdirectory(initialdir="d://", title="Choose Test files")
        self.ECTSetCycle.setEnabled(True)
        if datafilepath:
            fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3, figsize=(12, 6))
            tab = QtWidgets.QWidget()
            tab_layout = QtWidgets.QVBoxLayout(tab)
            canvas = FigureCanvas(fig)
            toolbar = NavigationToolbar(canvas, None)
            filecount = 0
            graphcolor = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
            if self.saveok.isChecked():
                save_file_name = filedialog.asksaveasfilename(initialdir="D://", title="Save File Name", defaultextension=".xlsx")
                if save_file_name:
                    writer = pd.ExcelWriter(save_file_name, engine="xlsxwriter")
            df = pd.DataFrame()
            predf = pd.DataFrame()
            # 하위 폴더 내의 파일까지 검색
            for (root, directories, files) in os.walk(datafilepath):
                for file in files:
                    if '.txt' in file:
                        file_path = os.path.join(root,file)
                        file_path = file_path.replace('\\','/')
                        subfile.append(file_path)
            # 폴더내의 파일 전체 취합
            for filename in subfile:
                predf = pd.read_csv(filename, on_bad_lines='skip', skiprows = 1)
                if predf.iloc[0,0] == "Time":
                    predf = pd.read_csv(filename, skiprows = 1, on_bad_lines='skip')
                predf = predf.iloc[:-1]
                df = pd.concat([df, predf], axis=0, ignore_index=True)
                filecountmax = len(subfile)
                progressdata = filecount/filecountmax * 100
                filecount = filecount + 1
                self.progressBar.setValue(int(progressdata))
            # 사이클을 위한 데이터 추출
            if not df.empty:
                df.columns = df.columns.str.replace('[^A-Za-z0-9_]+', '', regex=True)
                if "SC_VALUE" in df.columns:
                    if "LUT_VOLT0" in df.columns:
                        df = df[["Battery_Cycle", "SOH", "SOH_dR", "SOH_CA", "SOH_X", "LUT_VOLT0", "LUT_VOLT1", "LUT_VOLT2", "LUT_VOLT3",
                                 "SC_VALUE", "SC_SCORE", "SC_V_Acc", "SC_V_Avg"]]
                    else:
                        df = df[["Battery_Cycle", "SOH", "SOH_dR", "SOH_CA", "SOH_X", "SC_VALUE", "SC_SCORE", "SC_V_Acc", "SC_V_Avg"]]
                else:
                    df = df[["Battery_Cycle", "SOH", "SOH_dR", "SOH_CA", "SOH_X"]]
                # 중복 항목 제거 및 index reset
                df = df.drop_duplicates(subset="Battery_Cycle", keep='first', inplace=False, ignore_index=False)
                df = df.sort_values(by="Battery_Cycle")
                df.reset_index()
                # SOH 관련 단위 변경
                df["SOH"] = df["SOH"] / 10
                df["SOH_CA"] = df["SOH_CA"] / 10
                df["SOH_X"] = df["SOH_X"] / (-10)
                #그래프그리기
                graph_cycle(df["Battery_Cycle"], df["SOH"], ax1, 80, 105, 5, "Cycle", "Capacity ratio", "SOH", setxscale, graphcolor[0])
                graph_cycle(df["Battery_Cycle"], df["SOH_CA"], ax1, 80, 105, 5, "Cycle", "Capacity ratio", "SOH_CA", setxscale, graphcolor[1])
                graph_cycle(df["Battery_Cycle"], df["SOH_X"], ax4, -5, 15, 5, "Cycle", "Mass balance", "SOH_X", setxscale, graphcolor[3])
                graph_cycle(df["Battery_Cycle"], df["SOH_dR"], ax2, 0, 0, 0, "Cycle", "Anode Resistance", "SOH_dR", setxscale, graphcolor[2])
                if "SC_VALUE" in df.columns:
                    graph_cycle(df["Battery_Cycle"], df["SC_VALUE"], ax6, 0, 14, 2, "Cycle", "SC_VALUE", "SC_VALUE", setxscale, graphcolor[4])
                    graph_cycle(df["Battery_Cycle"], df["SC_SCORE"], ax3, 0, 6, 1, "Cycle", "SC_SCORE", "SC_SCORE", setxscale, graphcolor[5])
                    if "LUT_VOLT0" in df.columns:
                        graph_cycle(df["Battery_Cycle"], df["LUT_VOLT0"], ax5, 4, 4.6, 0.1, "Cycle", "Chg-Cut off", "1step limit",
                                    setxscale, graphcolor[0])
                        graph_cycle(df["Battery_Cycle"], df["LUT_VOLT1"], ax5, 4, 4.6, 0.1, "Cycle", "Chg-Cut off", "2step limit",
                                    setxscale, graphcolor[1])
                        graph_cycle(df["Battery_Cycle"], df["LUT_VOLT2"], ax5, 4, 4.6, 0.1, "Cycle", "Chg-Cut off", "3step limit",
                                    setxscale, graphcolor[2])
                        graph_cycle(df["Battery_Cycle"], df["LUT_VOLT3"], ax5, 4, 4.6, 0.1, "Cycle", "Chg-Cut off", "4step limit",
                                    setxscale, graphcolor[3])
                if self.saveok.isChecked() and save_file_name:
                    df.to_excel(writer, sheet_name="SETcycle")
                    writer.close()
            self.progressBar.setValue(100)
            tab_name_list = datafilepath.split("/")[-1].split(".")[-2]
            ax1.legend(loc="lower left")
            ax2.legend(loc="upper left")
            ax3.legend(loc="upper left")
            ax4.legend(loc="upper left")
            ax5.legend(loc="upper right")
            ax6.legend(loc="upper left")
            # fig.legend()
            tab_layout.addWidget(toolbar)
            tab_layout.addWidget(canvas)
            self.set_tab.addTab(tab, tab_name_list) 
            self.set_tab.setCurrentWidget(tab)
            plt.tight_layout(pad=1, w_pad=1, h_pad=1)
            output_fig(self.figsaveok, tab_name_list)
            plt.close()

    def ect_set_log_button(self):
        '''
        ECT result
        0 Time 1 voltage_now(mV) 2 Vavg(mV) 3 Ctype(Etc)-ChargCur 4 CurrentAvg. 5 Level
        6 Charging 7 Battery_Cycle 8 diffTime 9 Temperature(BA) 10 temp_ambient
        11 batt_temp 12 batt_temp_adc 13 batt_temp_raw 14 batt_wpc_temp 15 batt_wpc_temp_adc
        16 chg_temp 17 chg_temp_adc 18 dchg_temp 19 dchg_temp_adc 20 usb_temp 
        21 usb_temp_adc 22 compVoltage 23 ectSOC 24 RSOC 25 SOC_RE 
        26 SOC_EDV 27 SOH 28 AnodePotential 29 SOH_dR 30 SOH_CA 
        31 SOH_X 32 SC_VALUE 33 SC_SCORE 34 SC_V_Acc 35 SC_V_Avg 
        36 LUT_VOLT0 37 LUT_VOLT1 38 LUT_VOLT2 39 LUT_VOLT3 40 T_move
        '''
        global writer
        root = Tk()
        root.withdraw()
        self.ECTSetlog.setDisabled(True)
        datafilepaths = multi_askopendirnames()
        self.ECTSetlog.setEnabled(True)
        set_count = 1
        if datafilepaths:
            for set in datafilepaths:
                progressdata = progress(1, 1, 1, 1, set_count, len(datafilepaths))
                set_count = set_count + 1
                if self.saveok.isChecked():
                    save_file_name = filedialog.asksaveasfilename(initialdir="D://", title="Save File Name", defaultextension=".xlsx")
                    if save_file_name:
                        writer = pd.ExcelWriter(save_file_name, engine="xlsxwriter")
                    df = pd.DataFrame()
                fig, ax = plt.subplots(nrows=5, ncols=2, figsize=(18, 10))
                tab = QtWidgets.QWidget()
                tab_layout = QtWidgets.QVBoxLayout(tab)
                canvas = FigureCanvas(fig)
                toolbar = NavigationToolbar(canvas, None)
                appheader = ['Time', 'Vol', 'V avg (mV)', 'Ctype(Etc)-ChargCur', 'CurrAvg',
                            'SOC', 'fg_full_voltage', 'Charging', 'Battery_Cycle', 'sleepTime', 'diffTime',
                            'Temp', 'temp_ambient', 'batt_temp', 'batt_temp_adc', 'batt_temp_raw',
                            'batt_wpc_temp', 'batt_wpc_temp_adc', 'chg_temp', 'chg_temp_adc', 'dchg_temp',
                            'dchg_temp_adc', 'usb_temp', 'usb_temp_adc', 'SOCect', 'RSOC', 'SOC_RE', 'SOC_EDV',
                            'SOH', 'anodeE', 'SOH_dR', 'SOH_CA', 'SOH_X', 'SC_VALUE', 'SC_SCORE', 'SC_V_Acc',
                            'SC_V_Avg', 'LUT_VOLT0', 'LUT_VOLT1', 'LUT_VOLT2', 'LUT_VOLT3', 'T_move', 'OCV']
                inputheader = ['Time', 'log_SOC', 'log_Temp', 'log_Vol', 'log_CurrAvg', 'log_sleeptime', 'log_ect_full_voltage']
                outputheader = ['Time', 'log_ECT_CNT', 'log_SOCect', 'log_ECT_RSOC', 'log_ECT_SOC_RE', 'log_ECT_SOC_EDV', 'log_ECT_SOH', 'log_anodeE',
                                'log_ECT_SOH_dR', 'log_ECT_SOH_CA', 'log_ECT_SOH_X', 'log_ECT_ISD_Value', 'log_ECT_ISD_Score', 'log_ECT_ISD_Vacc',
                                'log_ECT_LDP_Step0', 'log_ECT_LDP_Step1', 'log_ECT_LDP_Step2', 'log_ECT_LDP_Step3', 'log_ECT_T_MOVE', 'log_ECT_OCV']
                # 파일 경로 패턴
                app_pattern = str(set) + '\\*ChemBatt_LOG*.txt'
                input_pattern = str(set) + '\\ect_inputlog.txt.*'
                output_pattern = str(set) + '\\ect_outputlog.txt.*'
                app_files = sorted(glob.glob(app_pattern))
                input_files = sorted(glob.glob(input_pattern))
                output_files = sorted(glob.glob(output_pattern))
                # 데이터프레임 로드 및 병합
                appdata = pd.concat(
                    (pd.read_csv(file, header=None, skiprows = 2, names=appheader) for file in app_files),
                    axis=0
                )
                inputlog = pd.concat(
                    (pd.read_csv(file, header=None, names=inputheader) for file in input_files),
                    axis=0
                )
                outputlog = pd.concat(
                    (pd.read_csv(file, header=None, names=outputheader) for file in output_files),
                    axis=0
                )
                # app결과의 시간을 timestamp로 수정
                appdata['Time'] = appdata['Time'].apply(to_timestamp)
                appdata['Time'] = appdata['Time'] + 10
                xlim = [inputlog["Time"].min() // 10000 * 10000, inputlog["Time"].max() // 10000 * 10000, 10000]
                inputlog.set_index('Time', inplace=True)
                outputlog.set_index('Time', inplace=True)
                appdata.set_index('Time', inplace=True)
                inputlog = inputlog[~inputlog.index.duplicated()]
                outputlog = outputlog[~outputlog.index.duplicated()]
                overall = pd.concat([inputlog, outputlog], axis=1, join='outer')
                # overall = overall[~overall.index.duplicated()]
                appdata = appdata[~appdata.index.duplicated()]
                overall = pd.concat([overall, appdata], axis=1, join='outer')
                overall['log_SOC'] = np.where(overall['log_SOC'] >= 10000, np.nan, overall['log_SOC'])
                # 충전 관련
                graph_set_profile(overall.index, overall.Vol/1000, ax[0, 0], 3.0, 4.8, 0.2, "Time(sec)", "Voltage (V)", "", 1, xlim[0], xlim[1], xlim[2])
                graph_set_profile(overall.index, overall.CurrAvg/1000, ax[1 ,0], 0, 0, 0, "Time(sec)", "CurrAvg (A)", "", 1, xlim[0], xlim[1], xlim[2])
                graph_set_profile(overall.index, overall.Temp, ax[2, 0], 20, 50, 4, "Time(sec)", "temp.(℃ )", "", 1, xlim[0], xlim[1], xlim[2])
                graph_set_profile(overall.index, overall.SOC, ax[3, 0], 0, 120, 10, "Time(sec)", "SOC", "", 1, xlim[0], xlim[1], xlim[2])
                # graph_set_profile(overall.index, overall.SOCect/ 10, ax[3, 0], 0, 120, 10, "Time(sec)", "SOC_ect", "", 1, xlim[0], xlim[1], xlim[2])
                graph_set_profile(overall.index, overall.anodeE/ 1000, ax[4, 0], -0.1, 1.6, 0.1, "Time(sec)", "anodeE (V)", "App", 1, xlim[0], xlim[1], xlim[2])
                graph_set_profile(overall.index, overall.log_Vol/1000, ax[0, 0], 3.0, 4.8, 0.2, "Time(sec)", "Voltage (V)", "", 2, xlim[0], xlim[1], xlim[2])
                graph_set_profile(overall.index, overall.log_CurrAvg/1000, ax[1, 0], 0, 0, 0, "Time(sec)", "CurrAvg (A)", "", 2, xlim[0], xlim[1], xlim[2])
                graph_set_profile(overall.index, overall.log_Temp/ 10, ax[2, 0], 20, 50, 4, "Time(sec)", "temp.(℃ )", "", 2, xlim[0], xlim[1], xlim[2])
                graph_set_profile(overall.index, overall.log_SOC/ 10, ax[3, 0], 0, 120, 10, "Time(sec)", "SOC", "", 2, xlim[0], xlim[1], xlim[2])
                # graph_set_profile(overall.index, overall.log_SOCect/ 10, ax[3, 0], 0, 120, 10, "Time(sec)", "SOC_ect", "", 2, xlim[0], xlim[1], xlim[2])
                graph_set_profile(overall.index, overall.log_anodeE/ 1000, ax[4, 0], -0.1, 1.6, 0.1, "Time(sec)", "anodeE (V)", "log", 2, xlim[0], xlim[1], xlim[2])
                graph_set_profile(overall.index, overall.Vol/1000 - overall.log_Vol/1000, ax[0, 1], -0.05, 0.06, 0.01, "Time(sec)", "Voltage (V)", "", 3, xlim[0], xlim[1], xlim[2])
                graph_set_profile(overall.index, overall.CurrAvg/1000 - overall.log_CurrAvg/1000, ax[1 ,1], -0.05, 0.06, 0.01, "Time(sec)", "CurrAvg (A)", "", 3, xlim[0], xlim[1], xlim[2])
                graph_set_profile(overall.index, overall.Temp - overall.log_Temp/10, ax[2, 1], -2, 3, 0.5, "Time(sec)", "temp.(℃ )", "", 3, xlim[0], xlim[1], xlim[2])
                graph_set_profile(overall.index, overall.SOC - overall.log_SOC/10, ax[3, 1], -5, 6, 1, "Time(sec)", "SOC", "", 3, xlim[0], xlim[1], xlim[2])
                # graph_set_profile(overall.index, overall.SOCect/ 10 - overall.log_SOCect/ 10, ax[4, 1], -5, 6, 1, "Time(sec)", "SOC_ect", "", 1, xlim[0], xlim[1], xlim[2])
                graph_set_profile(overall.index, overall.anodeE/ 1000 - overall.log_anodeE/ 1000, ax[4, 1], -0.1, 0.1, 0.02, "Time(sec)", "anodeE (V)", "delta", 3, xlim[0], xlim[1], xlim[2])
                plt.tight_layout(pad=1, w_pad=1, h_pad=1)
                for i in range(4):
                    for j in range(2):
                        # X축 레이블 제거
                        ax[i, j].set_xlabel('')
                        # X축 틱 레이블 제거
                        ax[i, j].set_xticklabels([])
                if self.saveok.isChecked() and save_file_name:
                    df = df._append(overall)
                if self.saveok.isChecked() and save_file_name:
                    if not self.chk_setcyc_sep.isChecked():
                        df.to_excel(writer, sheet_name="log")
                    else:
                        overall.to_excel(writer)
                    writer.close()
                fig.legend()
                plt.subplots_adjust(right=0.8)
                tab_name_list =set.split("/")[-1].split("\\")[-1]
                # plt.suptitle(Chgnamelist[-1], fontsize=15)
                tab_layout.addWidget(toolbar)
                tab_layout.addWidget(canvas)
                self.set_tab.addTab(tab, tab_name_list) 
                self.set_tab.setCurrentWidget(tab)
                plt.tight_layout(pad=1, w_pad=1, h_pad=1)
                output_fig(self.figsaveok, tab_name_list)
                plt.close()
            self.progressBar.setValue(100)

    def ect_set_log2_button(self):
        '''
        ECT result
        0 Time 1 voltage_now(mV) 2 Vavg(mV) 3 Ctype(Etc)-ChargCur 4 CurrentAvg. 5 Level
        6 Charging 7 Battery_Cycle 8 diffTime 9 Temperature(BA) 10 temp_ambient
        11 batt_temp 12 batt_temp_adc 13 batt_temp_raw 14 batt_wpc_temp 15 batt_wpc_temp_adc
        16 chg_temp 17 chg_temp_adc 18 dchg_temp 19 dchg_temp_adc 20 usb_temp 
        21 usb_temp_adc 22 compVoltage 23 ectSOC 24 RSOC 25 SOC_RE 
        26 SOC_EDV 27 SOH 28 AnodePotential 29 SOH_dR 30 SOH_CA 
        31 SOH_X 32 SC_VALUE 33 SC_SCORE 34 SC_V_Acc 35 SC_V_Avg 
        36 LUT_VOLT0 37 LUT_VOLT1 38 LUT_VOLT2 39 LUT_VOLT3 40 T_move
        '''
        global writer
        root = Tk()
        root.withdraw()
        self.ECTSetlog2.setDisabled(True)
        datafilepaths = multi_askopendirnames()
        self.ECTSetlog2.setEnabled(True)
        set_count = 1
        if self.saveok.isChecked():
            save_file_name = filedialog.asksaveasfilename(initialdir="D://", title="Save File Name")
        if datafilepaths:
            for set in datafilepaths:
                progressdata = progress(1, 1, 1, 1, set_count, len(datafilepaths))
                set_count = set_count + 1
                fig, ax = plt.subplots(nrows=5, ncols=2, figsize=(18, 10))
                tab = QtWidgets.QWidget()
                tab_layout = QtWidgets.QVBoxLayout(tab)
                canvas = FigureCanvas(fig)
                toolbar = NavigationToolbar(canvas, None)
                inputheader = ['Time', 'log_SOC', 'log_Temp', 'log_Vol', 'log_CurrAvg', 'log_sleeptime', 'log_ect_full_voltage', 'FG_cap']
                # curr_timestamp,
                # output.ECT_CNT, output.ECT_T_MOVE, output.ECT_OCV, output.ECT_ASOC, output.ECT_RSOC, 
                # output.ECT_SOC_RE, output.ECT_SOC_EDV, output.ECT_SOH, output.ECT_SOH_dR, output.ECT_SOH_CA, 
                # output.ECT_SOH_X, output.ECT_SOH_RSOH, output.ECT_ISD_Value, output.ECT_ISD_Score, output.ECT_ISD_Grade, output.ECT_ISD_Vacc, 
                # output.ECT_ISD_Vavg, output.ECT_ISD_I, output.ECT_ISD_R, output.ECT_LDP_0Step, output.ECT_LDP_1Step, output.ECT_LDP_2Step, 
                # output.ECT_LDP_3Step, output.ECT_Anode_Potential);
                outputheader0 = ['Time', 'log_ECT_CNT', 'log_ECT_T_MOVE', 'log_ECT_OCV', 'log_SOCect', 'log_ECT_RSOC',
                                'log_ECT_SOC_RE', 'log_ECT_SOC_EDV', 'log_ECT_SOH', 'log_ECT_SOH_dR', 'log_ECT_SOH_CA',
                                'log_ECT_SOH_X', 'log_ECT_SOH_MX', 'log_ECT_ISD_Value', 'log_ECT_ISD_Score', 'log_ECT_ISD_Vacc',
                                'log_ECT_ISD_Vavg','log_ECT_ISD_I', 'log_ECT_ISD_R', 'log_ECT_LDP_Step0', 'log_ECT_LDP_Step1', 'log_ECT_LDP_Step2',
                                'log_ECT_LDP_Step3', 'log_anodeE']
                outputheader = ['Time', 'log_ECT_CNT', 'log_ECT_T_MOVE', 'log_ECT_OCV', 'log_SOCect', 'log_ECT_RSOC',
                                'log_ECT_SOC_RE', 'log_ECT_SOC_EDV', 'log_ECT_SOH', 'log_ECT_SOH_dR', 'log_ECT_SOH_CA',
                                'log_ECT_SOH_X', 'log_ECT_SOH_MX', 'log_ECT_ISD_Value', 'log_ECT_ISD_Score', 'log_ECT_ISD_Grade', 'log_ECT_ISD_Vacc',
                                'log_ECT_ISD_Vavg','log_ECT_ISD_I', 'log_ECT_ISD_R', 'log_ECT_LDP_Step0', 'log_ECT_LDP_Step1', 'log_ECT_LDP_Step2',
                                'log_ECT_LDP_Step3', 'log_anodeE']
                # 파일 경로 패턴
                input_pattern = str(set) + '\\ect_inputlog.txt*'
                output_pattern = str(set) + '\\ect_outputlog.txt*'
                input_files = sorted(glob.glob(input_pattern))
                output_files = sorted(glob.glob(output_pattern))
                # 데이터프레임 로드 및 병합
                inputlog = pd.concat(
                    (pd.read_csv(file, header=None) for file in input_files),
                    axis=0
                )
                outputlog = pd.concat(
                    (pd.read_csv(file, header=None) for file in output_files),
                    axis=0
                )
                inputlog.columns = inputheader
                if len(outputlog.columns) == 24:
                    outputlog.columns = outputheader0
                else:
                    outputlog.columns = outputheader
                # app결과의 시간을 timestamp로 수정
                inputlog.set_index('Time', inplace=True)
                outputlog.set_index('Time', inplace=True)
                inputlog = inputlog[~inputlog.index.duplicated()]
                outputlog = outputlog[~outputlog.index.duplicated()]
                # x_min = 0
                # x_max = ((inputlog.index.max() - x_min) // 3600 * 3600) / 3600
                x_min = inputlog.index.min() // 3600 * 3600
                x_max = inputlog.index.max() // 3600 * 3600
                x_gap = 0
                # x_gap = (x_max - x_min) // 100
                # xlim = [x_min, x_max, x_gap]
                xlim = [0, (x_max - x_min) / 3600, x_gap]
                overall = pd.concat([inputlog, outputlog], axis=1, join='outer')
                overall['log_SOC'] = np.where(overall['log_SOC'] >= 10000, np.nan, overall['log_SOC'])
                # 충전 관련
                # graph_set_profile(overall.index, overall.log_Vol / 1000, ax[0, 0], 3.0, 4.8, 0.2, "Time(hr)", "Voltage (V)", "", 1, xlim[0], xlim[1], xlim[2])
                graph_set_profile((overall.index - x_min)/3600, overall.log_Vol / 1000, ax[0, 0], 3.0, 4.8, 0.2, "Time(hr)", "Voltage (V)", "", 1, xlim[0], xlim[1], xlim[2])
                graph_set_profile((overall.index - x_min)/3600, overall.log_ECT_LDP_Step0, ax[0, 0], 4.0, 4.8, 0.2, "Time(hr)", "Voltage (V)", "", 2, xlim[0], xlim[1], xlim[2])
                graph_set_profile((overall.index - x_min)/3600, overall.log_ECT_LDP_Step1, ax[0, 0], 4.0, 4.8, 0.2, "Time(hr)", "Voltage (V)", "", 3, xlim[0], xlim[1], xlim[2])
                graph_set_profile((overall.index - x_min)/3600, overall.log_ECT_LDP_Step2, ax[0, 0], 4.0, 4.8, 0.2, "Time(hr)", "Voltage (V)", "", 4, xlim[0], xlim[1], xlim[2])
                graph_set_profile((overall.index - x_min)/3600, overall.log_ECT_LDP_Step3, ax[0, 0], 4.0, 4.8, 0.2, "Time(hr)", "Voltage (V)", "", 5, xlim[0], xlim[1], xlim[2])
                graph_set_profile((overall.index - x_min)/3600, overall.log_ECT_OCV / 1000, ax[0, 0], 3.0, 4.8, 0.2, "Time(hr)", "Voltage (V)", "", 2, xlim[0], xlim[1], xlim[2])
                graph_set_profile((overall.index - x_min)/3600, overall.log_CurrAvg / 1000, ax[1, 0], 0, 0, 0, "Time(hr)", "CurrAvg/ISD (A)", "", 1, xlim[0], xlim[1], xlim[2])
                graph_set_profile((overall.index - x_min)/3600, overall.log_ECT_ISD_I / 1000, ax[1, 0], 0, 0, 0, "Time(hr)", "CurrAvg/ISD (A)", "", 2, xlim[0], xlim[1], xlim[2])
                graph_set_profile((overall.index - x_min)/3600, overall.log_Temp / 10, ax[2, 0], 20, 50, 4, "Time(hr)", "temp.(℃ )", "", 1, xlim[0], xlim[1], xlim[2])
                graph_set_profile((overall.index - x_min)/3600, overall.log_SOC / 10, ax[3, 0], 0, 120, 10, "Time(hr)", "SOC", "SOC", 1, xlim[0], xlim[1], xlim[2])
                graph_set_profile((overall.index - x_min)/3600, overall.log_SOCect / 10, ax[3, 0], 0, 120, 10, "Time(hr)", "SOC", "ECT_ASOC", 2, xlim[0], xlim[1], xlim[2])
                graph_set_profile((overall.index - x_min)/3600, overall.log_ECT_RSOC / 10, ax[3, 0], 0, 120, 10, "Time(hr)", "SOC", "ECT_RSOC", 3, xlim[0], xlim[1], xlim[2])
                graph_set_profile((overall.index - x_min)/3600, overall.log_ECT_SOC_RE / 10, ax[3, 0], 0, 120, 10, "Time(hr)", "SOC", "ECT_UISOC", 4, xlim[0], xlim[1], xlim[2])
                graph_set_profile((overall.index - x_min)/3600, overall.log_anodeE / 1000, ax[4, 0], -0.2, 1.6, 0.2, "Time(hr)", "anodeE (V)", "log", 1, xlim[0], xlim[1], xlim[2])
                graph_set_profile((overall.index - x_min)/3600, overall.log_ECT_SOH_dR, ax[0, 1], 0, 0, 0, "Time(hr)", "SEI resistance", "", 1, xlim[0], xlim[1], xlim[2])
                graph_set_profile((overall.index - x_min)/3600, overall.log_ECT_SOH / 10, ax[1, 1], 80, 110, 5, "Time(hr)", "SOH", "", 1, xlim[0], xlim[1], xlim[2])
                graph_set_profile((overall.index - x_min)/3600, overall.log_ECT_SOH_CA / 10, ax[1, 1], 80, 110, 5, "Time(hr)", "SOH", "", 2, xlim[0], xlim[1], xlim[2])
                graph_set_profile((overall.index - x_min)/3600, overall.log_ECT_SOH_X / 10, ax[2, 1], 0, 25, 5, "Time(hr)", "SOH_X", "", 1, xlim[0], xlim[1], xlim[2])
                graph_set_profile((overall.index - x_min)/3600, overall.log_ECT_ISD_Value, ax[3, 1], 0, 11, 1, "Time(hr)", "ISD value", "", 1, xlim[0], xlim[1], xlim[2])
                graph_set_profile((overall.index - x_min)/3600, overall.log_ECT_ISD_Score, ax[3, 1], 0, 6, 1, "Time(hr)", "ISD score", "", 2, xlim[0], xlim[1], xlim[2])
                graph_set_profile((overall.index - x_min)/3600, overall.log_ECT_ISD_R, ax[4, 1], 0, 1100, 100, "Time(hr)", "ISD resistance", "", 1, xlim[0], xlim[1], xlim[2])
                plt.tight_layout(pad=1, w_pad=1, h_pad=1)
                tab_name_list =set.split("/")[-1].split("\\")[-1]
                for i in range(4):
                    for j in range(2):
                        # X축 레이블 제거
                        ax[i, j].set_xlabel('')
                        # X축 틱 레이블 제거
                        ax[i, j].set_xticklabels([])
                if self.saveok.isChecked() and save_file_name:
                    overall_sorted = overall.sort_index()
                    overall_sorted.to_csv(save_file_name + '_' + tab_name_list + '.csv')
                fig.legend()
                plt.subplots_adjust(right=0.8)
                # plt.suptitle(Chgnamelist[-1], fontsize=15)
                tab_layout.addWidget(toolbar)
                tab_layout.addWidget(canvas)
                self.set_tab.addTab(tab, tab_name_list) 
                self.set_tab.setCurrentWidget(tab)
                plt.tight_layout(pad=1, w_pad=1, h_pad=1)
                output_fig(self.figsaveok, tab_name_list)
                plt.close()
            self.progressBar.setValue(100)

    # 소재 정보 데이터 파일 경로 설정
    def dvdq_material_button(self):
        ca_mat_filepath = filedialog.askopenfilename(initialdir="d://dvdqraw//", title="양극 개별 Profile 데이터 선택")
        self.ca_mat_dvdq_path.setText(str(ca_mat_filepath))
        an_mat_filepath = filedialog.askopenfilename(initialdir="d://dvdqraw//", title="음극 개별 Profile 데이터 선택")
        self.an_mat_dvdq_path.setText(str(an_mat_filepath))

    def dvdq_profile_button(self):
        real_filepath = filedialog.askopenfilename(initialdir="d://dvdqraw//", title="실측 결과 Profile 데이터 선택")
        self.pro_dvdq_path.setText(str(real_filepath))

    def dvdq_ini_reset_button(self):
        self.ca_mass_ini.setText(str(""))
        self.ca_slip_ini.setText(str(""))
        self.an_mass_ini.setText(str(""))
        self.an_slip_ini.setText(str(""))
        self.fittingdegree = 1
        self.min_rms = np.inf

    def dvdq_graph(self, simul_full, min_params, rms):
        self.tab_delete(self.dvdq_simul_tab)
        # while self.dvdq_simul_tab.count() > 0:
        #     self.dvdq_simul_tab.removeTab(0)
        # tab 그래프 추가
        fig = plt.figure(figsize=(8, 8))
        tab = QtWidgets.QWidget()
        tab_layout = QtWidgets.QVBoxLayout(tab)
        canvas = FigureCanvas(fig)
        ax1 = plt.subplot(2, 1, 1)
        ax2 = plt.subplot(2, 1, 2)
        toolbar = NavigationToolbar(canvas, None)
        # Voltage Profile 그리기
        ax1.plot(simul_full.full_cap, simul_full.an_volt, "-", color = "b")
        ax1.plot(simul_full.full_cap, simul_full.ca_volt, "-", color = "r")
        ax1.plot(simul_full.full_cap, simul_full.full_volt, "--", color = "g")
        ax1.plot(simul_full.full_cap, simul_full.real_volt, "-", color = "k")
        ax1.set_ylim(0, 4.7)
        ax1.set_xticks(np.linspace(-5, 105, 23))
        ax1.legend(["음극", "양극", "예측", "실측"])
        ax1.set_xlabel("SOC")
        ax1.set_ylabel("Voltage")
        ax1.grid(which="major", axis="both", alpha=0.5)
        # dVdQ 그래프 그리기
        ax2.plot(simul_full.full_cap, simul_full.an_dvdq, "-", color = "b")
        ax2.plot(simul_full.full_cap, simul_full.ca_dvdq, "-", color = "r")
        ax2.plot(simul_full.full_cap, simul_full.full_dvdq, "--", color = "g")
        ax2.plot(simul_full.full_cap, simul_full.real_dvdq, "-", color = "k")
        ax2.set_ylim(-0.02, 0.02)
        ax2.set_xticks(np.linspace(-5, 105, 23))
        ax2.legend(["음극 dVdQ", "양극 dVdQ", "예측 dVdQ", "실측 dVdQ"])
        ax2.set_xlabel("SOC")
        ax2.set_ylabel("dVdQ")
        ax2.grid(which="major", axis="both", alpha=0.5)
        fig.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
        # chtname = real_filepath.split("/")
        fig.legend()
        # SaveFileName set-up
        fig.suptitle(
            # 'title:' + str(chtname[-1])
            'ca_mass:' + str(f"{min_params[0]:.2f}")
            + ',     ca_slip:' + str(f"{min_params[1]:.2f}")
            + ',     an_mass:' + str(f"{min_params[2]:.2f}")
            + ',     an_slip:' + str(f"{min_params[3]:.2f}")
            + ',     rms(%):' + str(f"{(rms * 100):.3f}")
            , fontsize= 12)
        tab_layout.addWidget(toolbar)
        tab_layout.addWidget(canvas)
        self.dvdq_simul_tab.addTab(tab, str(self.pro_dvdq_path.text()))
        self.dvdq_simul_tab.setCurrentWidget(tab)
        plt.tight_layout(pad=1, w_pad=1, h_pad=1)
        plt.close()

    def dvdq_fitting_button(self):
        global writer
        ca_mat_filepath = str(self.ca_mat_dvdq_path.text())
        an_mat_filepath = str(self.an_mat_dvdq_path.text())
        real_filepath = str(self.pro_dvdq_path.text())
        if "mAh" in real_filepath: # 파일 이름에 용량 관련 문자 있을 때
                dvdq_min_cap = name_capacity(real_filepath)
        else:
            dvdq_min_cap = 100
        if os.path.isfile(ca_mat_filepath):
            ca_ccv_raw = pd.read_csv(ca_mat_filepath, sep="\t")
            an_ccv_raw = pd.read_csv(an_mat_filepath, sep="\t")
        else:
            self.dvdq_material_button()
            ca_mat_filepath = str(self.ca_mat_dvdq_path.text())
            an_mat_filepath = str(self.an_mat_dvdq_path.text())
            ca_ccv_raw = pd.read_csv(ca_mat_filepath, sep="\t")
            an_ccv_raw = pd.read_csv(an_mat_filepath, sep="\t")
        ca_ccv_raw.columns = ["ca_cap", "ca_volt"]
        an_ccv_raw.columns = ["an_cap", "an_volt"]
        if os.path.isfile(real_filepath):
            real_raw = pd.read_csv(real_filepath, sep="\t")
        else:
            self.dvdq_profile_button()
            real_filepath = str(self.pro_dvdq_path.text())
            real_raw = pd.read_csv(real_filepath, sep="\t")
        real_raw.columns = ["real_cap", "real_volt"]
        # 셀 용량 기준
        full_cell_max_cap = max(real_raw.real_cap)
        # 미분을 위한 smoothing 기준
        full_period = int(self.dvdq_full_smoothing_no.text())
        # 각 parameter의 초기값 세팅
        if self.ca_mass_ini.text() == "":
            ca_mass_ini = full_cell_max_cap/max(ca_ccv_raw.ca_cap)
            self.ca_mass_ini.setText(str(ca_mass_ini))
        else:
            ca_mass_ini = float(self.ca_mass_ini.text())
        if self.ca_slip_ini.text() == "":
            ca_slip_ini = 1
            self.ca_slip_ini.setText(str(ca_slip_ini))
        else:
            ca_slip_ini = float(self.ca_slip_ini.text())
        if self.an_mass_ini.text() == "":
            an_mass_ini = full_cell_max_cap/max(an_ccv_raw.an_cap)
            self.an_mass_ini.setText(str(an_mass_ini))
        else:
            an_mass_ini = float(self.an_mass_ini.text())
        if self.an_slip_ini.text() == "":
            an_slip_ini = 1
            self.an_slip_ini.setText(str(an_slip_ini))
        else:
            an_slip_ini = float(self.an_slip_ini.text())
        # 열화 상태 고려 및 LL 산포에 따른 보정치 추가
        self.fittingdegree = self.fittingdegree * 1.2
        if self.ca_mass_ini_fix.isChecked():
            ca_mass_min = ca_mass_ini
            ca_mass_max = ca_mass_ini
        else:
            ca_mass_min = ca_mass_ini * (1 - 0.1 / self.fittingdegree)
            ca_mass_max = ca_mass_ini * (1 + 0.1 / self.fittingdegree)
        if self.ca_slip_ini_fix.isChecked():
            ca_slip_min = ca_slip_ini
            ca_slip_max = ca_slip_ini
        else:
            ca_slip_min = ca_slip_ini - (full_cell_max_cap * (0.05 / self.fittingdegree))
            ca_slip_max = ca_slip_ini + (full_cell_max_cap * (0.05 / self.fittingdegree))
        if self.an_mass_ini_fix.isChecked():
            an_mass_min = an_mass_ini
            an_mass_max = an_mass_ini
        else:
            an_mass_min = an_mass_ini * (1 - 0.1 / self.fittingdegree)
            an_mass_max = an_mass_ini * (1 + 0.1 / self.fittingdegree)
        if self.an_slip_ini_fix.isChecked():
            an_slip_min = an_slip_ini
            an_slip_max = an_slip_ini
        else:
            an_slip_min = an_slip_ini - (full_cell_max_cap * (0.05 / self.fittingdegree))
            an_slip_max = an_slip_ini + (full_cell_max_cap * (0.05 / self.fittingdegree))
        # 목적함수 초기화 실행
        min_params = None
        for i in range(int(self.dvdq_test_no.text())):
            ca_mass, ca_slip, an_mass, an_slip = generate_params(ca_mass_min, ca_mass_max, ca_slip_min, ca_slip_max, an_mass_min, an_mass_max,
                                                                 an_slip_min, an_slip_max)
            simul_full = generate_simulation_full(ca_ccv_raw, an_ccv_raw, real_raw, ca_mass, ca_slip, an_mass, an_slip, full_cell_max_cap,
                                                  dvdq_min_cap, full_period)
            # 지정 영역에서만 rms 산정
            simul_full = simul_full.loc[(simul_full["full_cap"] > int(
                self.dvdq_start_soc.text())) & (simul_full["full_cap"] < int(self.dvdq_end_soc.text()))]
            simul_diff = np.subtract(simul_full.full_dvdq, simul_full.real_dvdq)
            simul_diff[np.isnan(simul_diff)] = 0
            simul_rms = np.sqrt(np.mean(simul_diff ** 2))
            if simul_rms < self.min_rms:
                self.min_rms = simul_rms
                min_params = (ca_mass, ca_slip, an_mass, an_slip)
                self.dvdq_rms.setText(str(self.min_rms * 100))
                self.ca_mass_ini.setText(str(min_params[0]))
                self.ca_slip_ini.setText(str(min_params[1]))
                self.an_mass_ini.setText(str(min_params[2]))
                self.an_slip_ini.setText(str(min_params[3]))
            self.progressBar.setValue(int(int(i)/int(self.dvdq_test_no.text())*100))
        if min_params is not None:
            simul_full = generate_simulation_full(ca_ccv_raw, an_ccv_raw, real_raw, min_params[0], min_params[1],
                                                    min_params[2], min_params[3], full_cell_max_cap, dvdq_min_cap, full_period)
            simul_full = simul_full.loc[(simul_full["full_cap"] > int(
                self.dvdq_start_soc.text())) & (simul_full["full_cap"] < int(self.dvdq_end_soc.text()))]
            self.dvdq_graph(simul_full, min_params, self.min_rms)
            ca_max_cap = max(ca_ccv_raw.ca_cap) * min_params[0]
            an_max_cap = max(an_ccv_raw.an_cap) * min_params[2]
            self.full_cell_max_cap_txt.setText(str(full_cell_max_cap))
            self.ca_max_cap_txt.setText(str(ca_max_cap))
            self.an_max_cap_txt.setText(str(an_max_cap))
            result_para = pd.DataFrame({"para": min_params})
            if self.saveok.isChecked():
                save_file_name = filedialog.asksaveasfilename(initialdir="D://", title="Save File Name", defaultextension=".xlsx")
                if save_file_name != '':
                    # parameter를 별도 시트에 저장
                    writer = pd.ExcelWriter(save_file_name, engine="xlsxwriter")
                    result_para.to_excel(writer, sheet_name="parameter", index=False)
                    simul_full.to_excel(writer, sheet_name="dvdq", index=False)
                    writer.close()
        else:
            self.dvdq_fitting2_button()
        self.progressBar.setValue(100)

    def dvdq_fitting2_button(self):
        self.fittingdegree = 1
        self.min_rms = np.inf
        if self.ca_mass_ini.text():
            global writer
            ca_mat_filepath = str(self.ca_mat_dvdq_path.text())
            an_mat_filepath = str(self.an_mat_dvdq_path.text())
            real_filepath = str(self.pro_dvdq_path.text())
            if "mAh" in real_filepath: # 파일 이름에 용량 관련 문자 있을 때
                dvdq_min_cap = name_capacity(real_filepath)
            else:
                dvdq_min_cap = 100
            ca_ccv_raw = pd.read_csv(ca_mat_filepath, sep="\t")
            ca_ccv_raw.columns = ["ca_cap", "ca_volt"]
            an_ccv_raw = pd.read_csv(an_mat_filepath, sep="\t")
            an_ccv_raw.columns = ["an_cap", "an_volt"]
            real_raw = pd.read_csv(real_filepath, sep="\t")
            real_raw.columns = ["real_cap", "real_volt"]
            
            # 셀 용량 기준
            full_cell_max_cap = max(real_raw.real_cap)
            # 미분을 위한 smoothing 기준
            full_period = int(self.dvdq_full_smoothing_no.text())
            # 각 parameter의 초기값 세팅
            ca_mass_ini = float(self.ca_mass_ini.text())
            ca_slip_ini = float(self.ca_slip_ini.text())
            an_mass_ini = float(self.an_mass_ini.text())
            an_slip_ini = float(self.an_slip_ini.text())
            simul_full = generate_simulation_full(ca_ccv_raw, an_ccv_raw, real_raw, ca_mass_ini, ca_slip_ini,
                                                  an_mass_ini, an_slip_ini, full_cell_max_cap, dvdq_min_cap, full_period)
            simul_full = simul_full.loc[(simul_full["full_cap"] > int(self.dvdq_start_soc.text())) &
                                        (simul_full["full_cap"] < int(self.dvdq_end_soc.text()))]
            simul_diff = np.subtract(simul_full.full_dvdq, simul_full.real_dvdq)
            simul_diff[np.isnan(simul_diff)] = 0
            simul_rms = np.sqrt(np.mean(simul_diff ** 2))
            self.min_rms = simul_rms
            self.dvdq_rms.setText(str(self.min_rms * 100))
            ini_params = [ca_mass_ini, ca_slip_ini, an_mass_ini, an_slip_ini]
            self.dvdq_graph(simul_full, ini_params, simul_rms)
            self.progressBar.setValue(100)
            ca_max_cap = max(ca_ccv_raw.ca_cap) * ca_mass_ini
            an_max_cap = max(an_ccv_raw.an_cap) * an_mass_ini
            self.full_cell_max_cap_txt.setText(str(full_cell_max_cap))
            self.ca_max_cap_txt.setText(str(ca_max_cap))
            self.an_max_cap_txt.setText(str(an_max_cap))
            result_para = pd.DataFrame({"para": [ca_mass_ini, ca_slip_ini, an_mass_ini, an_slip_ini]})
            if self.saveok.isChecked():
                save_file_name = filedialog.asksaveasfilename(initialdir="D://", title="Save File Name", defaultextension=".xlsx")
                writer = pd.ExcelWriter(save_file_name, engine="xlsxwriter")
                if save_file_name:
                    # parameter를 별도 시트에 저장
                    result_para.to_excel(writer, sheet_name="parameter", index=False)
                    simul_full.to_excel(writer, sheet_name="dvdq", index=False)
                    writer.close()
    
    def load_cycparameter_button(self):
        cyc_filepaths = filedialog.askopenfilenames(initialdir="d://cycparameter//",
                                                    title="불러올 02C, 05C 사이클 parameter 데이터를 순차적으로 선택")
        self.cycparameter.setText(str(cyc_filepaths[0]))
        parameterfilepath = self.cycparameter.text()
        self.cycparameter2.setText(str(cyc_filepaths[1]))
        parameterfilepath2 = self.cycparameter2.text()
        self.folderappcycestimation.setEnabled(True)
        self.pathappcycestimation.setEnabled(True)
        if parameterfilepath:
            parameter_df = pd.read_csv(parameterfilepath, sep="\t", engine="c", encoding="UTF-8", skiprows=1, on_bad_lines='skip')
            parameter_df = parameter_df.dropna(axis=0)
            self.aTextEdit_02c.setText(str(round(parameter_df.iloc[0, 0], 50)))
            self.bTextEdit_02c.setText(str(round(parameter_df.iloc[1, 0], 50)))
            self.b1TextEdit_02c.setText(str(round(parameter_df.iloc[2, 0], 50)))
            self.cTextEdit_02c.setText(str(round(parameter_df.iloc[3, 0], 50)))
            self.dTextEdit_02c.setText(str(round(parameter_df.iloc[4, 0], 50)))
            self.eTextEdit_02c.setText(str(round(parameter_df.iloc[5, 0], 50)))
            self.fTextEdit_02c.setText(str(round(parameter_df.iloc[6, 0], 50)))
            self.fdTextEdit_02c.setText(str(round(parameter_df.iloc[7, 0], 50)))
            
        if parameterfilepath2:
            parameter_df2 = pd.read_csv(parameterfilepath2, sep="\t", engine="c", encoding="UTF-8", skiprows=1, on_bad_lines='skip')
            parameter_df2 = parameter_df2.dropna(axis=0)
            self.aTextEdit_05c.setText(str(round(parameter_df2.iloc[0, 0], 50)))
            self.bTextEdit_05c.setText(str(round(parameter_df2.iloc[1, 0], 50)))
            self.b1TextEdit_05c.setText(str(round(parameter_df2.iloc[2, 0], 50)))
            self.cTextEdit_05c.setText(str(round(parameter_df2.iloc[3, 0], 50)))
            self.dTextEdit_05c.setText(str(round(parameter_df2.iloc[4, 0], 50)))
            self.eTextEdit_05c.setText(str(round(parameter_df2.iloc[5, 0], 50)))
            self.fTextEdit_05c.setText(str(round(parameter_df2.iloc[6, 0], 50)))
            self.fdTextEdit_05c.setText(str(round(parameter_df2.iloc[7, 0], 50)))
    
    def app_cycle_tab_reset_button(self):
        self.tab_delete(self.cycle_simul_tab)
        self.tab_no = 0

    def path_approval_cycle_estimation_button(self):
        def cyccapparameter(x, f_d):
            return 1 - np.exp(a_par1 * x[1] + b_par1) * (x[0] * f_d) ** b1_par1 - np.exp(c_par1 * x[1] + d_par1) * (
                x[0] * f_d) ** (e_par1 * x[1] + f_par1)
        def cyccapparameter02(x, f_d):
            return 1 - np.exp(a_par2 * x[1] + b_par2) * (x[0] * f_d) ** b1_par2 - np.exp(c_par2 * x[1] + d_par2) * (
                x[0] * f_d) ** (e_par2 * x[1] + f_par2)
        a_par1 = float(self.aTextEdit_02c.text())
        b_par1 = float(self.bTextEdit_02c.text())
        b1_par1 = float(self.b1TextEdit_02c.text())
        c_par1 = float(self.cTextEdit_02c.text())
        d_par1 = float(self.dTextEdit_02c.text())
        e_par1 = float(self.eTextEdit_02c.text())
        f_par1 = float(self.fTextEdit_02c.text())
        fd_par1 = float(self.fdTextEdit_02c.text())
        
        a_par2 = float(self.aTextEdit_05c.text())
        b_par2 = float(self.bTextEdit_05c.text())
        b1_par2 = float(self.b1TextEdit_05c.text())
        c_par2 = float(self.cTextEdit_05c.text())
        d_par2 = float(self.dTextEdit_05c.text())
        e_par2 = float(self.eTextEdit_05c.text())
        f_par2 = float(self.fTextEdit_05c.text())
        fd_par2 = float(self.fdTextEdit_05c.text())
        
        firstCrate, mincapacity, xscale, ylimithigh, ylimitlow, irscale = self.cyc_ini_set()
        # 용량 선정 관련
        global writer
        foldercount, chnlcount, colorno, writecolno = 0, 0, 0, 0
        root = Tk()
        root.withdraw()
        self.pathappcycestimation.setDisabled(True)
        self.chk_cyclepath.setChecked(True)
        pne_path = self.pne_path_setting()
        self.chk_cyclepath.setChecked(False)
        all_data_folder = pne_path[0]
        all_data_name = pne_path[1]
        if self.saveok.isChecked():
            save_file_name = filedialog.asksaveasfilename(initialdir="D://", title="Save File Name", defaultextension=".xlsx")
            if save_file_name:
                writer = pd.ExcelWriter(save_file_name, engine="xlsxwriter")
        self.pathappcycestimation.setEnabled(True)
        graphcolor = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        for i, cyclefolder in enumerate(all_data_folder):
            # tab 그래프 추가
            fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
            if os.path.exists(cyclefolder):
                if os.path.isdir(cyclefolder):
                    subfolder = [f.path for f in os.scandir(cyclefolder) if f.is_dir()]
                    foldercountmax = len(all_data_folder)
                    foldercount = foldercount + 1
                    for FolderBase in subfolder:
                        # tab 그래프 추가
                        tab = QtWidgets.QWidget()
                        tab_layout = QtWidgets.QVBoxLayout(tab)
                        canvas = FigureCanvas(fig)
                        toolbar = NavigationToolbar(canvas, None)
                        chnlcountmax = len(subfolder)
                        chnlcount = chnlcount + 1
                        # progressdata = (foldercount + chnlcount/chnlcountmax - 1)/foldercountmax * 100
                        progressdata = progress(foldercount, foldercountmax, chnlcount, chnlcountmax, 1, 1)
                        self.progressBar.setValue(int(progressdata))
                        cycnamelist = FolderBase.split("\\")
                        headername = [cycnamelist[-2] + ", " + cycnamelist[-1]]
                        if len(all_data_name) != 0:
                            title = all_data_name[i]
                        else:
                            title = cycnamelist[-2]
                        if not check_cycler(cyclefolder):
                            pass
                        else:
                            cyctemp = pne_simul_cycle_data(FolderBase, mincapacity, firstCrate)
                        # print(FolderBase)
                        if isinstance(cyctemp[1], pd.DataFrame) and hasattr(cyctemp[1], "Dchg"):
                            self.capacitytext.setText(str(cyctemp[0]))
                            if self.cyc_long_life.isChecked() and hasattr(cyctemp[1], "long_acc"):
                                cyctemp[1]["Dchg"] = cyctemp[1]["Dchg"] - cyctemp[1]["long_acc"]
                            y1 = cyctemp[1]["Dchg"]
                            t1 =cyctemp[1]["Temp"]
                            y2 = cyctemp[3]["Dchg"]
                            t2 =cyctemp[3]["Temp"]
                            if t1.max() < 273:
                                t1 = 23 + 273
                                t2 = 23 + 273
                            x1 = cyctemp[1].index
                            x2 = cyctemp[3].index
                            dataadd1 = pd.DataFrame({'x1': x1, 't1': t1, 'y1': y1})
                            dataadd2 = pd.DataFrame({'x2': x2, 't2': t2, 'y2': y2})
                            dfall1 = dataadd1.dropna()
                            dfall2 = dataadd2.dropna()
                            maxfevset = 5000
                            p1 = [fd_par1]
                            p2 = [fd_par2]
                            if self.simul_x_max.text() == 0:
                                df1 = pd.DataFrame({'x1': range(1, 2000, 1)})
                                df2 = pd.DataFrame({'x2': range(1, 2000, 1)})
                            else:
                                df1 = pd.DataFrame({'x1': range(1, int(self.simul_x_max.text()), 1)})
                                df2 = pd.DataFrame({'x2': range(1, int(self.simul_x_max.text()), 1)})
                            df1.t1 = 23 + 273
                            df2.t2 = 23 + 273
                            popt1, pcov1 = curve_fit(cyccapparameter, (dfall1.x1, dfall1.t1), dfall1.y1, p1, maxfev=maxfevset)
                            popt2, pcov2 = curve_fit(cyccapparameter02, (dfall2.x2, dfall2.t2), dfall2.y2, p2, maxfev=maxfevset)
                            residuals1 = dfall1.y1 - cyccapparameter((dfall1.x1, dfall1.t1), *popt1)
                            residuals2 = dfall2.y2 - cyccapparameter02((dfall2.x2, dfall2.t2), *popt2)
                            ss_res1 = np.sum(residuals1 ** 2)
                            ss_res2 = np.sum(residuals2 ** 2)
                            ss_tot1 = np.sum((dfall1.y1 - np.mean(dfall1.y1)) ** 2)
                            ss_tot2 = np.sum((dfall2.y2 - np.mean(dfall2.y2)) ** 2)
                            r_squared1 = 1 - (ss_res1/ss_tot1)
                            r_squared2 = 1 - (ss_res2/ss_tot2)
                            if self.simul_long_life.isChecked():
                                real_y1 = dfall1.y1 * cyctemp[2] - cyctemp[1]["long_acc"]
                                simul_y1 = cyccapparameter((df1.x1, df1.t1), *popt1) * cyctemp[2] - cyctemp[1]["long_acc"]
                            else:
                                real_y1 = dfall1.y1 * cyctemp[2]
                                simul_y1 = cyccapparameter((df1.x1, df1.t1), *popt1) * cyctemp[2]
                            ax1.plot(dfall1.x1, real_y1, 'o', color=graphcolor[colorno], markersize=3,
                                     label='가속 = %5.3f' % tuple(popt1[0] / p1))
                            ax1.plot(df1.x1, simul_y1, color=graphcolor[colorno], label='오차 = %5.3f' % r_squared1)
                            ax1.plot(dfall2.x2, dfall2.y2 * cyctemp[4], 'o', color=graphcolor[colorno], markersize=5,
                                     label='가속 = %5.3f' % tuple(popt2[0] / p2))
                            ax1.plot(df2.x2, cyccapparameter02((df2.x2, df2.t2), *popt2) * cyctemp[4], '--', color=graphcolor[colorno],
                                     label='오차 = %5.3f' % r_squared2)
                            colorno = colorno % 9 + 1
                            # Data output option
                            output_df_all = cyctemp[7][["Dchg", "Temp", "Curr", "max_vol", "min_vol"]]
                            output_df_05c = cyctemp[1][["Dchg", "Temp", "Curr", "max_vol", "min_vol"]]
                            output_df_02c = cyctemp[3][["Dchg", "Temp", "Curr", "max_vol", "min_vol"]]
                            if self.saveok.isChecked() and save_file_name:
                                output_df_all.to_excel(writer, sheet_name="app_cycle", startcol=writecolno)
                                output_df_05c.to_excel(writer, sheet_name="highrate_cycle", startcol=writecolno)
                                output_df_02c.to_excel(writer, sheet_name="rate02c_cycle", startcol=writecolno)
                                writecolno = writecolno + 6
                            if len(all_data_name) != 0:
                                plt.suptitle(title, fontsize= 15, fontweight='bold')
                            else:
                                plt.suptitle(title, fontsize= 15, fontweight='bold')
                            ax1.tick_params(axis='both', which='major', labelsize=12) 
                            ax1.legend(loc="center left", bbox_to_anchor=(1, 0.5))
                            ax1.set_ylim(float(self.simul_y_min.text()), float(self.simul_y_max.text()))
                            # ax1.set_xlim(0.5, 1.1)
                            ax1.set_ylabel('capacity ratio', fontsize = 14)
                            ax1.set_xlabel('cycle or day', fontsize = 14)
                            ax1.grid(which="major", axis="both", alpha=.5)
                            plt.tight_layout(pad=1, w_pad=1, h_pad=1)
                            output_fig(self.figsaveok, cycnamelist[-2])
                    tab_layout.addWidget(toolbar)
                    tab_layout.addWidget(canvas)
                    self.cycle_simul_tab.addTab(tab, f"예측{i+1}")
                    self.cycle_simul_tab.setCurrentWidget(tab)
                    plt.tight_layout(pad=1, w_pad=1, h_pad=1)
        # plt.show()
        if self.saveok.isChecked() and save_file_name:
            writer.close()
        plt.tight_layout(pad=1, w_pad=1, h_pad=1)
        plt.close()
        self.progressBar.setValue(100)
    
    def folder_approval_cycle_estimation_button(self):
        def cyccapparameter(x, f_d):
            return 1 - np.exp(a_par1 * x[1] + b_par1) * (x[0] * f_d) ** b1_par1 - np.exp(c_par1 * x[1] + d_par1) \
                * (x[0] * f_d) ** (e_par1 * x[1] + f_par1)
        def cyccapparameter02(x, f_d):
            return 1 - np.exp(a_par2 * x[1] + b_par2) * (x[0] * f_d) ** b1_par2 - np.exp(c_par2 * x[1] + d_par2) \
                * (x[0] * f_d) ** (e_par2 * x[1] + f_par2)
        
        a_par1 = float(self.aTextEdit_02c.text())
        b_par1 = float(self.bTextEdit_02c.text())
        b1_par1 = float(self.b1TextEdit_02c.text())
        c_par1 = float(self.cTextEdit_02c.text())
        d_par1 = float(self.dTextEdit_02c.text())
        e_par1 = float(self.eTextEdit_02c.text())
        f_par1 = float(self.fTextEdit_02c.text())
        fd_par1 = float(self.fdTextEdit_02c.text())
        
        a_par2 = float(self.aTextEdit_05c.text())
        b_par2 = float(self.bTextEdit_05c.text())
        b1_par2 = float(self.b1TextEdit_05c.text())
        c_par2 = float(self.cTextEdit_05c.text())
        d_par2 = float(self.dTextEdit_05c.text())
        e_par2 = float(self.eTextEdit_05c.text())
        f_par2 = float(self.fTextEdit_05c.text())
        fd_par2 = float(self.fdTextEdit_05c.text())
        
        firstCrate, mincapacity, xscale, ylimithigh, ylimitlow, irscale = self.cyc_ini_set()
        # 용량 선정 관련
        global writer
        foldercount, chnlcount, colorno, num = 0, 0, 0, 0
        root = Tk()
        root.withdraw()
        self.folderappcycestimation.setDisabled(True)
        pne_path = filedialog.askopenfilenames(initialdir="D://", title="Data File Name", defaultextension=".txt")
        self.folderappcycestimation.setEnabled(True)
        graphcolor = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        for i, cyclefolder in enumerate(pne_path):
            # tab 그래프 추가
            if os.path.exists(cyclefolder):
                foldercountmax = len(pne_path)
                foldercount = foldercount + 1
                # tab 그래프 추가
                chnlcountmax = len(cyclefolder)
                chnlcount = chnlcount + 1
                # progressdata = (foldercount + chnlcount/chnlcountmax - 1)/foldercountmax * 100
                progressdata = progress(foldercount, foldercountmax, chnlcount, chnlcountmax, 1, 1)
                self.progressBar.setValue(int(progressdata))
                title = cyclefolder
                df = pd.read_csv(cyclefolder, sep="\t", engine="c", encoding="UTF-8", skiprows=1, on_bad_lines='skip')
                for i in range(0, len(df.columns)):
                    if i % 6 == 5:
                        df_trim = df.iloc[:, (i - 5): (i + 1)]
                        df_trim.columns = ['TotlCycle', 'Dchg', 'Temp', 'Curr', 'max_vol', 'min_vol']
                        cyctemp = pne_simul_cycle_data_file(df_trim, cyclefolder, mincapacity, firstCrate)
                        if isinstance(cyctemp[1], pd.DataFrame) and hasattr(cyctemp[1], "Dchg"):
                            num = num + 1
                            fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(12, 12))
                            tab = QtWidgets.QWidget()
                            tab_layout = QtWidgets.QVBoxLayout(tab)
                            canvas = FigureCanvas(fig)
                            toolbar = NavigationToolbar(canvas, None)
                            self.capacitytext.setText(str(cyctemp[0]))
                            if self.cyc_long_life.isChecked() and hasattr(cyctemp[1], "long_acc"):
                                cyctemp[1]["Dchg"] = cyctemp[1]["Dchg"] - cyctemp[1]["long_acc"]
                            y1 = cyctemp[1]["Dchg"]
                            t1 = cyctemp[1]["Temp"]
                            y2 = cyctemp[3]["Dchg"]
                            t2 = cyctemp[3]["Temp"]
                            if t1.max() < 273:
                                t1 = 23 + 273
                                t2 = 23 + 273
                            x1 = cyctemp[1].index
                            x2 = cyctemp[3].index
                            dataadd1 = pd.DataFrame({'x1': x1, 't1': t1, 'y1': y1})
                            dataadd2 = pd.DataFrame({'x2': x2, 't2': t2, 'y2': y2})
                            dfall1 = dataadd1.dropna()
                            dfall2 = dataadd2.dropna()
                            maxfevset = 5000
                            p1 = [fd_par1]
                            p2 = [fd_par2]
                            if self.simul_x_max.text() == 0:
                                df1 = pd.DataFrame({'x1': range(1, 2000, 1)})
                                df2 = pd.DataFrame({'x2': range(1, 2000, 1)})
                            else:
                                df1 = pd.DataFrame({'x1': range(1, int(self.simul_x_max.text()), 1)})
                                df2 = pd.DataFrame({'x2': range(1, int(self.simul_x_max.text()), 1)})
                            df1.t1 = 23 + 273
                            df2.t2 = 23 + 273
                            popt1, pcov1 = curve_fit(cyccapparameter, (dfall1.x1, dfall1.t1), dfall1.y1, p1, maxfev=maxfevset)
                            popt2, pcov2 = curve_fit(cyccapparameter02, (dfall2.x2, dfall2.t2), dfall2.y2, p2, maxfev=maxfevset)
                            residuals1 = dfall1.y1 - cyccapparameter((dfall1.x1, dfall1.t1), *popt1)
                            residuals2 = dfall2.y2 - cyccapparameter02((dfall2.x2, dfall2.t2), *popt2)
                            ss_res1 = np.sum(residuals1 ** 2)
                            ss_res2 = np.sum(residuals2 ** 2)
                            ss_tot1 = np.sum((dfall1.y1 - np.mean(dfall1.y1)) ** 2)
                            ss_tot2 = np.sum((dfall2.y2 - np.mean(dfall2.y2)) ** 2)
                            r_squared1 = 1 - (ss_res1/ss_tot1)
                            r_squared2 = 1 - (ss_res2/ss_tot2)
                            if self.simul_long_life.isChecked():
                                real_y1 = dfall1.y1 * cyctemp[2] - cyctemp[1]["long_acc"]
                                simul_y1 = cyccapparameter((df1.x1, df1.t1), *popt1) * cyctemp[2] - cyctemp[1]["long_acc"]
                            else:
                                real_y1 = dfall1.y1 * cyctemp[2]
                                simul_y1 = cyccapparameter((df1.x1, df1.t1), *popt1) * cyctemp[2]
                            ax1.plot(dfall1.x1, real_y1, 'o', color=graphcolor[colorno], markersize=3,
                                     label='사이클 가속 = %5.3f' % tuple(popt1[0] / p1))
                            ax1.plot(df1.x1, simul_y1, color=graphcolor[colorno], label='사이클 오차 = %5.3f' % r_squared1)
                            ax1.plot(dfall2.x2, dfall2.y2 * cyctemp[4], 'o', color=graphcolor[colorno], markersize=3,
                                     label='0.2C 가속 = %5.3f' % tuple(popt2[0] / p2))
                            ax1.plot(df2.x2, cyccapparameter02((df2.x2, df2.t2), *popt2) * cyctemp[4], '--', color=graphcolor[colorno],
                                     label='0.2C 오차 = %5.3f' % r_squared2)
                            colorno = colorno % 9 + 1
                            ax1.tick_params(axis='both', which='major', labelsize=12) 
                            ax1.legend(loc="center left", bbox_to_anchor=(1, 0.5))
                            ax1.set_ylim(float(self.simul_y_min.text()), float(self.simul_y_max.text()))
                            ax1.set_ylabel('capacity ratio', fontsize = 14)
                            ax1.set_xlabel('cycle or day', fontsize = 14)
                            ax1.grid(which="major", axis="both", alpha=.5)
                            plt.tight_layout(pad=1, w_pad=1, h_pad=1)
                            plt.suptitle(title, fontsize= 15, fontweight='bold')
                            tab_layout.addWidget(toolbar)
                            tab_layout.addWidget(canvas)
                            self.cycle_simul_tab.addTab(tab, f"예측{num}")
                            self.cycle_simul_tab.setCurrentWidget(tab)
                            plt.tight_layout(pad=1, w_pad=1, h_pad=1)
        plt.tight_layout(pad=1, w_pad=1, h_pad=1)
        plt.close()
        self.progressBar.setValue(100)

    def eu_tab_reset_button(self):
        self.tab_delete(self.cycle_simul_tab_eu)
        self.tab_no = 0
        
    def eu_parameter_reset_button(self):
        # 초기 매개변수 리스트 (이름과 순서를 명시적으로 정의)
        initial_params = {
            "a": 0.03,
            "b": -18,
            "b1": 0.7,
            "c": 2.3,
            "d": -782,
            "e": -0.28,
            "f": 96,
            "fd": 1
        }
        # 각 매개변수를 텍스트 필드에 설정
        for param_name, value in initial_params.items():
            # QTextEdit 객체 이름 패턴: self.{param_name}TextEdit_eu
            widget_name = f"{param_name}TextEdit_eu"
            # 객체가 존재할 경우 값 설정
            if hasattr(self, widget_name):
                getattr(self, widget_name).setText(f"{value:.5f}")  # 소수점 5자리까지 표시

    def eu_load_cycparameter_button(self):
        # 파일 선택 대화상자 열기
        cyc_filepath = filedialog.askopenfilename(initialdir=r"d://", title="사이클 파라미터 데이터 선택")
        self.cycparameter_eu.setText(cyc_filepath)
        parameterfilepath = self.cycparameter_eu.text()

        if parameterfilepath:
            # CSV 파일 읽기 (오류 발생 시 NaN으로 처리)
            parameter_df = pd.read_csv(parameterfilepath, sep="\t", engine="c", encoding="UTF-8",
                                       skiprows=1, on_bad_lines='skip')
            # 결측치 제거
            parameter_df = parameter_df.dropna(axis=0)
            # "02C" 컬럼 존재 여부 확인
            if "02C" in parameter_df.columns:
                # "02C" 컬럼 데이터 추출
                col_data = parameter_df["02C"]
                
                # 각 매개변수를 텍스트박스에 설정 (과학적 표기법 방지)
                self.aTextEdit_eu.setText(f"{col_data.iloc[0]:.15f}")
                self.bTextEdit_eu.setText(f"{col_data.iloc[1]:.15f}")
                self.b1TextEdit_eu.setText(f"{col_data.iloc[2]:.15f}")
                self.cTextEdit_eu.setText(f"{col_data.iloc[3]:.15f}")
                self.dTextEdit_eu.setText(f"{col_data.iloc[4]:.15f}")
                self.eTextEdit_eu.setText(f"{col_data.iloc[5]:.15f}")
                self.fTextEdit_eu.setText(f"{col_data.iloc[6]:.15f}")
                self.fdTextEdit_eu.setText(f"{col_data.iloc[7]:.15f}")
            else:
                # "02C" 컬럼이 없는 경우 경고 메시지 출력
                err_msg("에러","parameter 컬럼 없음") 
                # self.aTextEdit_eu.setText("02C 컬럼 없음")
    
    def eu_save_cycparameter_button(self):
        par1 = float(self.aTextEdit_eu.text())
        par2 = float(self.bTextEdit_eu.text())
        par3 = float(self.b1TextEdit_eu.text())
        par4 = float(self.cTextEdit_eu.text())
        par5 = float(self.dTextEdit_eu.text())
        par6 = float(self.eTextEdit_eu.text())
        par7 = float(self.fTextEdit_eu.text())
        par8 = float(self.fdTextEdit_eu.text())
        filepath = self.cycparameter_eu.text()
        namelist = filepath.split("/")
        namelist = namelist[-1].split(".")
        
        df = pd.DataFrame()
        df["02C"] = [par1, par2, par3, par4, par5, par6, par7, par8]
        save_file_name = "d:/para_" + namelist[0] + ".txt"
        # parameter를 별도 시트에 저장
        with open(save_file_name, 'w') as f:
            f.write('\n')
            df.to_csv(f, sep='\t', index=False, header=True)
        err_msg("저장", "저장되었습니다.") 
        output_para_fig(self.figsaveok, "fig_" + namelist[0])

    def eu_fitting_confirm_button(self):
        global writer
        # exp complex degradation (cycle 기준, day 기준)
        def swellingfit(x, a_par, b_par, b1_par, c_par, d_par, e_par, f_par, f_d):
            return np.exp(a_par * x[1] + b_par) * (x[0] * f_d) ** b1_par + np.exp(c_par * x[1] + d_par) * (x[0] * f_d) ** (e_par * x[1] + f_par)
        def capacityfit(x, a_par, b_par, b1_par, c_par, d_par, e_par, f_par, f_d):
            return 1 - np.exp(a_par * x[1] + b_par) * (x[0] * f_d) ** b1_par - np.exp(c_par * x[1] + d_par) * (x[0] * f_d) ** (e_par * x[1] + f_par)
        if self.aTextEdit_eu.text() == "":
            parini1 = [0.03, -18, 0.7, 2.3, -782, -0.28, 96, 1]
            self.aTextEdit_eu.setText(str(parini1[0]))
            self.bTextEdit_eu.setText(str(parini1[1]))
            self.b1TextEdit_eu.setText(str(parini1[2]))
            self.cTextEdit_eu.setText(str(parini1[3]))
            self.dTextEdit_eu.setText(str(parini1[4]))
            self.eTextEdit_eu.setText(str(parini1[5]))
            self.fTextEdit_eu.setText(str(parini1[6]))
            self.fdTextEdit_eu.setText(str(parini1[7]))
        # UI 변수를 초기치로 산정
        a_par1 = float(self.aTextEdit_eu.text())
        b_par1 = float(self.bTextEdit_eu.text())
        b1_par1 = float(self.b1TextEdit_eu.text())
        c_par1 = float(self.cTextEdit_eu.text())
        d_par1 = float(self.dTextEdit_eu.text())
        e_par1 = float(self.eTextEdit_eu.text())
        f_par1 = float(self.fTextEdit_eu.text())
        fd_par1 = float(self.fdTextEdit_eu.text())
        p0 = [a_par1, b_par1, b1_par1, c_par1, d_par1, e_par1, f_par1, fd_par1]
        root = Tk()
        root.withdraw()
        self.xscale = int(self.simul_x_max_eu.text())
        y_max = float(self.simul_y_max_eu.text())
        y_min = float(self.simul_y_min_eu.text())
        datafilepath = filedialog.askopenfilenames(initialdir="d://", title="Choose Test files")
        if datafilepath:
            self.cycparameter_eu.setText(str(datafilepath[0]))
        file = 0
        num = 0
        if self.saveok.isChecked():
            save_file_name = filedialog.asksaveasfilename(initialdir="D://", title="Save File Name", defaultextension=".xlsx")
            if save_file_name:
                writer = pd.ExcelWriter(save_file_name, engine="xlsxwriter")
        for filepath in datafilepath:
            # tab 그래프 추가
            fig, ax = plt.subplots(figsize=(6, 6))
            tab = QtWidgets.QWidget()
            tab_layout = QtWidgets.QVBoxLayout(tab)
            canvas = FigureCanvas(fig)
            toolbar = NavigationToolbar(canvas, None)
            dfall = pd.DataFrame({'x': [], 't': [], 'y': []})
            df = pd.read_csv(filepath, sep="\t", engine="c", encoding="UTF-8", skiprows=1, on_bad_lines='skip')
            if not "02C" in df.columns:
                namelist = filepath.split("/")
                namelist = namelist[-1].split(".")
                filemax = len(datafilepath)
                file = file + 1
                first_column_name = df.columns[1]
                if isinstance(first_column_name, str) and first_column_name.isdigit():
                    for i in range(1, len(df.columns)):
                        y1 = df.iloc[:,i]
                        x1 = df.iloc[:len(y1),0]
                        # # 초기값으로 100분율 환산
                        dataadd = pd.DataFrame({'x': x1, 'y': y1})
                        dataadd = dataadd.dropna()
                        if df.iloc[0, i] != 0 and not dataadd.empty and len(dataadd) >= 2 and dataadd.shape[1] >= 2:
                            x_0 = dataadd.iloc[0,0]
                            x_1 = dataadd.iloc[1,0]
                            y_0 = dataadd.iloc[0,1]
                            y_1 = dataadd.iloc[1,1]
                            coefficients = np.polyfit([x_0, x_1], [y_0, y_1], 1)
                            polynomial = np.poly1d(coefficients)
                            y_max_cap = polynomial(0)
                            if not self.fix_swelling_eu.isChecked():
                                dataadd.y = dataadd.y/y_max_cap
                        dataadd['t'] = float(df.columns[i][:2])
                        if dataadd.t.max() < 273:
                            dataadd.t = dataadd.t + 273
                        # 전체 dataframe을 누적
                        dfall = dfall._append(dataadd)
                        progressdata = (file + 1 + num/len(list(df)))/filemax * 100
                        self.progressBar.setValue(int(progressdata))
                else:
                    for i in range(0, len(df.columns)):
                        num = num + 1
                        if i % 3 == 2:
                            y1 = df.iloc[:, i]
                            x1 = df.iloc[:, i - 2][:len(y1)]
                            # # 초기값으로 100분율 환산
                            if df.iloc[0, i] != 0:
                                x_0 = x1[0]
                                x_1 = x1[1]
                                y_0 = y1[0]
                                y_1 = y1[1]
                                coefficients = np.polyfit([x_0, x_1], [y_0, y_1], 1)
                                polynomial = np.poly1d(coefficients)
                                y_max_cap = polynomial(0)
                                if not self.fix_swelling_eu.isChecked():
                                    y1 = y1/y_max_cap
                            # 용량이 있는 값을 기준으로 온도를 절대값으로 환산
                            t1 = df.iloc[:, i - 1][:len(y1)]
                            if t1.max() < 273:
                                t1 = t1 + 273
                            # 용량이 있는 값을 기준으로 cycle 산정
                            dataadd = pd.DataFrame({'x': x1, 't': t1, 'y': y1})
                            # 전체 dataframe을 누적
                            dfall = dfall._append(dataadd)
                            progressdata = (file + 1 + num/len(list(df)))/filemax * 100
                            self.progressBar.setValue(int(progressdata))
                dfall = dfall.dropna()
                # 초기값을 100%로 맞추는 데이터 추가
                df2 = pd.DataFrame({'x': range(1, self.xscale, 1)})
                if self.fix_swelling_eu.isChecked():
                    df2.t0 = 23 + 273
                    df2.t1 = 35 + 273
                    df2.t2 = 40 + 273
                    df2.t3 = 45 + 273
                else:
                    df2.t1 = 23 + 273
                    df2.t2 = 35 + 273
                    df2.t3 = 45 + 273
                # Fitting 계산
                maxfevset = 100000
                try:
                    if self.fix_swelling_eu.isChecked():
                        popt, pcov = curve_fit(swellingfit, (dfall.x, dfall.t), dfall.y, p0, maxfev=maxfevset)
                    else:
                        popt, pcov = curve_fit(capacityfit, (dfall.x, dfall.t), dfall.y, p0, maxfev=maxfevset)
                except (RuntimeError, TypeError) as e:
                    continue
                if self.fix_swelling_eu.isChecked():
                    residuals = dfall.y - swellingfit((dfall.x, dfall.t), *popt)
                else:
                    residuals = dfall.y - capacityfit((dfall.x, dfall.t), *popt)
                ss_res = np.sum(residuals ** 2)
                ss_tot = np.sum((dfall.y - np.mean(dfall.y)) ** 2)
                r_squared = 1 - (ss_res/ss_tot)
                if self.fix_swelling_eu.isChecked():
                    result0 = swellingfit((df2.x, df2.t0), *popt)
                    result1 = swellingfit((df2.x, df2.t1), *popt)
                    result2 = swellingfit((df2.x, df2.t2), *popt)
                    result3 = swellingfit((df2.x, df2.t3), *popt)
                    eol_cycle = result1.abs().idxmin()
                else:
                    result1 = capacityfit((df2.x, df2.t1), *popt)
                    result2 = capacityfit((df2.x, df2.t2), *popt)
                    result3 = capacityfit((df2.x, df2.t3), *popt)
                    eol_cycle = (result1 - 0.8).abs().idxmin()
                self.aTextEdit_eu.setText(str(round(popt[0], 50)))
                self.bTextEdit_eu.setText(str(round(popt[1], 50)))
                self.b1TextEdit_eu.setText(str(round(popt[2], 50)))
                self.cTextEdit_eu.setText(str(round(popt[3], 50)))
                self.dTextEdit_eu.setText(str(round(popt[4], 50)))
                self.eTextEdit_eu.setText(str(round(popt[5], 50)))
                self.fTextEdit_eu.setText(str(round(popt[6], 50)))
                self.fdTextEdit_eu.setText(str(round(popt[7], 50)))
                if self.fix_swelling_eu.isChecked():
                    dfall_23 = dfall[dfall["t"] == 296]
                    dfall_35 = dfall[dfall["t"] == 308]
                    dfall_40 = dfall[dfall["t"] == 313]
                    dfall_45 = dfall[dfall["t"] == 318]
                    ax.plot(df2.x, result0, 'blue', label='23도')
                    ax.plot(df2.x, result1, 'orange', label='35도')
                    ax.plot(df2.x, result2, 'pink', label='40도/ R^2 = %5.4f' % r_squared)
                    ax.plot(df2.x, result3, 'red', label='45도/' + namelist[-2])
                    ax.plot(dfall_23.x, dfall_23.y, 'o', color='blue', markersize=2)
                    ax.plot(dfall_35.x, dfall_35.y, 'o', color='orange', markersize=2)
                    ax.plot(dfall_40.x, dfall_40.y, 'o', color='pink', markersize=2)
                    ax.plot(dfall_45.x, dfall_45.y, 'o', color='red', markersize=2)
                else:
                    dfall_23 = dfall[dfall["t"] == 296]
                    dfall_35 = dfall[dfall["t"] == 308]
                    dfall_45 = dfall[dfall["t"] == 318]
                    ax.plot(df2.x, result1, 'blue', label='23도/R^2 = %5.4f' % r_squared)
                    ax.plot(df2.x, result2, 'orange', label='35도/' + namelist[-2])
                    ax.plot(df2.x, result3, 'red', label='45도/' + str(eol_cycle))
                    ax.plot(dfall_23.x, dfall_23.y, 'o', color='blue', markersize=2)
                    ax.plot(dfall_35.x, dfall_35.y, 'o', color='orange', markersize=2)
                    ax.plot(dfall_45.x, dfall_45.y, 'o', color='red', markersize=2)
                graph_eu_set(ax, y_min, y_max)
                plt.suptitle('fit: a=%5.4f, b=%5.4f, b1=%5.4f, c=%5.4f, d=%5.4f, e=%5.4f, f=%5.4f, f_d=%5.4f' % tuple(popt), fontsize= 14)
                plt.tight_layout(pad=1, w_pad=1, h_pad=1)
                # 데이터 저장
                result0 = df2.x
                if self.fix_swelling_eu.isChecked():
                    result = pd.DataFrame({"x": result0, str(df2.t1 - 273): result1, "40": result2, "45": result3})
                else:
                    result = pd.DataFrame({"x": result0, str(df2.t1 - 273): result1, "35": result2, "45": result3})
                result_para = pd.DataFrame({"para": popt})
                if self.saveok.isChecked() and save_file_name:
                    result_para.to_excel(writer, sheet_name="parameter", index=False)
                    result.to_excel(writer, sheet_name=str(namelist[-2][:30]), index=False)
                tab_layout.addWidget(toolbar)
                tab_layout.addWidget(canvas)
                filename =filepath.split(".t")[-2].split("/")[-1].split("\\")[-1]
                self.cycle_simul_tab_eu.addTab(tab, filename)
                self.cycle_simul_tab_eu.setCurrentWidget(tab)
                plt.tight_layout(pad=1, w_pad=1, h_pad=1)
        if self.saveok.isChecked() and save_file_name:
            writer.close()
        plt.tight_layout(pad=1, w_pad=1, h_pad=1)
        self.progressBar.setValue(100)
        plt.close()
    
    def eu_constant_fitting_confirm_button(self):
        global writer
        # exp 열화 모드 - parameter 고정 후 가속 계수 확인
        def cyccapparameter(x, f_d):
            return 1 - np.exp(a_par1 * x[1] + b_par1) * (x[0] * f_d) ** b1_par1 - np.exp(c_par1 * x[1] + d_par1) * (
                x[0] * f_d) ** (e_par1 * x[1] + f_par1)
        def cycswellingparameter(x, f_d):
            return np.exp(a_par1 * x[1] + b_par1) * (x[0] * f_d) ** b1_par1 + np.exp(c_par1 * x[1] + d_par1) * (
                x[0] * f_d) ** (e_par1 * x[1] + f_par1)
        
        if self.aTextEdit_eu.text() == "":
            parini1 = [0.03, -18, 0.7, 2.3, -782, -0.28, 96, 1]
            self.aTextEdit_eu.setText(str(parini1[0]))
            self.bTextEdit_eu.setText(str(parini1[1]))
            self.b1TextEdit_eu.setText(str(parini1[2]))
            self.cTextEdit_eu.setText(str(parini1[3]))
            self.dTextEdit_eu.setText(str(parini1[4]))
            self.eTextEdit_eu.setText(str(parini1[5]))
            self.fTextEdit_eu.setText(str(parini1[6]))
            self.fdTextEdit_eu.setText(str(parini1[7]))
        
        # UI 변수를 초기치로 산정
        a_par1 = float(self.aTextEdit_eu.text())
        b_par1 = float(self.bTextEdit_eu.text())
        b1_par1 = float(self.b1TextEdit_eu.text())
        c_par1 = float(self.cTextEdit_eu.text())
        d_par1 = float(self.dTextEdit_eu.text())
        e_par1 = float(self.eTextEdit_eu.text())
        f_par1 = float(self.fTextEdit_eu.text())
        fd_par1 = float(self.fdTextEdit_eu.text())
        self.xscale = int(self.simul_x_max_eu.text())
        y_max = float(self.simul_y_max_eu.text())
        y_min = float(self.simul_y_min_eu.text())
        # 결과 온도 관련
        temp = [23, 28, 35, 40, 45]
        # parameter 계산 및 fd 산정
        root = Tk()
        root.withdraw()
        datafilepath = filedialog.askopenfilenames(initialdir="d://", title="Choose Test files")
        if datafilepath:
            self.cycparameter_eu.setText(str(datafilepath[0]))
        file = 0
        num = 0
        writerowno = 0
        result = pd.DataFrame()
        raw_all = pd.DataFrame()
        if self.saveok.isChecked():
            save_file_name = filedialog.asksaveasfilename(initialdir="D://", title="Save File Name", defaultextension=".xlsx")
            if save_file_name:
                writer = pd.ExcelWriter(save_file_name, engine="xlsxwriter")
        # 3열 데이터 연결, 절대온도 변환, max 1로 변경
        for filepath in datafilepath:
            # tab 그래프 추가
            fig, ax = plt.subplots(figsize=(6, 6))
            tab = QtWidgets.QWidget()
            tab_layout = QtWidgets.QVBoxLayout(tab)
            canvas = FigureCanvas(fig)
            toolbar = NavigationToolbar(canvas, None)
            dfall = pd.DataFrame({'x': [], 't': [], 'y': []})
            df = pd.read_csv(filepath, sep="\t", engine="c", encoding="UTF-8", skiprows=1, on_bad_lines='skip')
            if not "02C" in df.columns:
                const_namelist = filepath.split("/")
                filemax = len(datafilepath)
                file = file + 1
                first_column_name = df.columns[1]
                if isinstance(first_column_name, str) and first_column_name.isdigit():
                    for i in range(1, len(df.columns)):
                        y1 = df.iloc[:,i]
                        x1 = df.iloc[:len(y1),0]
                        # 초기값으로 100분율 환산
                        dataadd = pd.DataFrame({'x': x1, 'y': y1})
                        dataadd = dataadd.dropna()
                        if df.iloc[0, i] != 0:
                            x_0 = dataadd.iloc[0,0]
                            x_1 = dataadd.iloc[1,0]
                            y_0 = dataadd.iloc[0,1]
                            y_1 = dataadd.iloc[1,1]
                            coefficients = np.polyfit([x_0, x_1], [y_0, y_1], 1)
                            polynomial = np.poly1d(coefficients)
                            y_max_cap = polynomial(1)
                            if not self.fix_swelling_eu.isChecked():
                                dataadd.y = dataadd.y/y_max_cap
                        raw_all = pd.concat([dataadd, raw_all], axis=1)
                        dataadd['t'] = float(df.columns[i][:2])
                        if dataadd.t.max() < 273:
                            dataadd.t = dataadd.t + 273
                        # 전체 dataframe을 누적
                        dfall = dfall._append(dataadd)
                        progressdata = (file + 1 + num/len(list(df)))/filemax * 100
                        self.progressBar.setValue(int(progressdata))
                else:
                    for i in range(0, len(df.columns)):
                        num = num + 1
                        if i % 3 == 2:
                            y1 = df.iloc[:, i]
                            x1 = df.iloc[:, i - 2][:len(y1)]
                            # if df.iloc[0, i] != 0:
                            #     y1 = y1/df.iloc[0, i]
                            # 초기값으로 100분율 환산
                            if df.iloc[0, i] != 0:
                                x_0 = x1[0]
                                x_1 = x1[1]
                                y_0 = y1[0]
                                y_1 = y1[1]
                                coefficients = np.polyfit([x_0, x_1], [y_0, y_1], 1)
                                polynomial = np.poly1d(coefficients)
                                y_max_cap = polynomial(1)
                                if not self.fix_swelling_eu.isChecked():
                                    y1 = y1/y_max_cap
                            t1 = df.iloc[:, i - 1][:len(y1)]
                            if t1.max() < 273:
                                t1 = t1 + 273
                            dataadd = pd.DataFrame({'x': x1, 't': t1, 'y': y1})
                            raw_all = pd.concat([dfall, raw_all], axis=1)
                            dfall = dfall._append(dataadd)
                            progressdata = (file + 1 + num/len(list(df)))/filemax * 100
                            self.progressBar.setValue(int(progressdata))
                dfall = dfall.dropna()
                maxfevset = 5000
                p0 = [fd_par1]
                df2 = pd.DataFrame({'x': range(1, self.xscale, 1)})
                # if self.fix_swelling_eu.isChecked():
                df2.t1 = temp[0] + 273
                df2.t2 = temp[1] + 273
                df2.t3 = temp[2] + 273
                df2.t4 = temp[3] + 273
                df2.t5 = temp[4] + 273
                try:
                    if self.fix_swelling_eu.isChecked():
                        popt, pcov = curve_fit(cycswellingparameter, (dfall.x, dfall.t), dfall.y, p0, maxfev=maxfevset)
                    else:
                        popt, pcov = curve_fit(cyccapparameter, (dfall.x, dfall.t), dfall.y, p0, maxfev=maxfevset)
                except (RuntimeError, TypeError) as e:
                    continue
                if self.fix_swelling_eu.isChecked():
                    residuals = dfall.y -cycswellingparameter((dfall.x, dfall.t), *popt)
                else:
                    residuals = dfall.y -cyccapparameter((dfall.x, dfall.t), *popt)
                ss_res = np.sum(residuals ** 2)
                ss_tot = np.sum((dfall.y - np.mean(dfall.y)) ** 2)
                r_squared = 1 - (ss_res/ss_tot)
                if self.fix_swelling_eu.isChecked():
                    result1 = cycswellingparameter((df2.x, df2.t1), *popt)
                    result2 = cycswellingparameter((df2.x, df2.t2), *popt)
                    result3 = cycswellingparameter((df2.x, df2.t3), *popt)
                    result4 = cycswellingparameter((df2.x, df2.t4), *popt)
                    result5 = cycswellingparameter((df2.x, df2.t5), *popt)
                    eol_cycle1 = (0.08 - result1).abs().idxmin()
                    eol_cycle2 = (0.08 - result2).abs().idxmin()
                    eol_cycle3 = (0.08 - result3).abs().idxmin()
                    eol_cycle4 = (0.08 - result4).abs().idxmin()
                    eol_cycle5 = (0.08 - result5).abs().idxmin()
                else:
                    result1 = cyccapparameter((df2.x, df2.t1), *popt)
                    result2 = cyccapparameter((df2.x, df2.t2), *popt)
                    result3 = cyccapparameter((df2.x, df2.t3), *popt)
                    result4 = cyccapparameter((df2.x, df2.t4), *popt)
                    result5 = cyccapparameter((df2.x, df2.t5), *popt)
                    eol_cycle1 = (result1 - 0.8).abs().idxmin()
                    eol_cycle2 = (result2 - 0.8).abs().idxmin()
                    eol_cycle3 = (result3 - 0.8).abs().idxmin()
                    eol_cycle4 = (result4 - 0.8).abs().idxmin()
                    eol_cycle5 = (result5 - 0.8).abs().idxmin()
                r_square_label = 'R^2 = %5.4f' % r_squared
                self.fdTextEdit_eu_2.setText(str(round(popt[0], 50)))
                filename = filepath.split(".t")[-2].split("/")[-1].split("\\")[-1]
                dfall_1 = dfall[dfall["t"] == temp[0] + 273]
                dfall_2 = dfall[dfall["t"] == temp[1] + 273]
                dfall_3 = dfall[dfall["t"] == temp[2] + 273]
                dfall_4 = dfall[dfall["t"] == temp[3] + 273]
                dfall_5 = dfall[dfall["t"] == temp[4] + 273]
                ax.plot(df2.x, result1, 'b-', label= str(temp[0]) + ' / ' + str(eol_cycle1))
                ax.plot(df2.x, result2, 'g-', label= str(temp[1]) + ' / ' + str(eol_cycle2))
                ax.plot(df2.x, result3, 'orange', label= str(temp[2]) + ' / ' + str(eol_cycle3))
                ax.plot(df2.x, result4, 'pink', label= str(temp[3]) + ' / ' + str(eol_cycle4) + '/R^2 = %5.4f' % r_squared)
                ax.plot(df2.x, result5, 'r-', label= str(temp[4]) + ' / ' + str(eol_cycle5) + '/가속 = %5.3f' % tuple(popt[0] / p0))
                if not dfall_1.empty:
                    ax.plot(dfall_1.x, dfall_1.y, 'o', color='blue', markersize=1)
                if not dfall_2.empty:
                    ax.plot(dfall_2.x, dfall_2.y, 'o', color='green', markersize=1)
                if not dfall_3.empty:
                    ax.plot(dfall_3.x, dfall_3.y, 'o', color='orange', markersize=1)
                if not dfall_4.empty:
                    ax.plot(dfall_4.x, dfall_4.y, 'o', color='pink', markersize=1)
                if not dfall_5.empty:
                    ax.plot(dfall_5.x, dfall_5.y, 'o', color='red', markersize=1)
                graph_eu_set(ax, y_min, y_max)
                plt.suptitle(filename, fontsize= 14, fontweight='bold')
                plt.tight_layout(pad=1, w_pad=1, h_pad=1)
                # 데이터 저장
                result0 = df2.x
                # if self.fix_swelling_eu.isChecked():
                result = pd.DataFrame({"x": result0, str(df2.t1 - 273): result1, str(df2.t2 - 273): result2, str(df2.t3 - 273): result3,
                                        str(df2.t4 - 273): result4, str(df2.t5 - 273): result5})
                result_para = pd.DataFrame({"para": popt})
                if self.saveok.isChecked() and save_file_name:
                    output_data(result_para, "parameter", writerowno + 1, 1, "para", [const_namelist[-1]])
                    writerowno = writerowno + 1
                    result.to_excel(writer, sheet_name="estimation", index=False)
                    raw_all.to_excel(writer, sheet_name="raw", index=False)
                tab_layout.addWidget(toolbar)
                tab_layout.addWidget(canvas)
                self.cycle_simul_tab_eu.addTab(tab, filename)
                self.cycle_simul_tab_eu.setCurrentWidget(tab)
                plt.tight_layout(pad=1, w_pad=1, h_pad=1)
        if self.saveok.isChecked() and save_file_name:
            writer.close()
        plt.tight_layout(pad=1, w_pad=1, h_pad=1)
        self.progressBar.setValue(100)
        plt.close()

    def eu_indiv_constant_fitting_confirm_button(self):
        global writer
        # exp 열화 모드 - parameter 고정 후 가속 계수 확인
        def cyccapparameter(x, f_d):
            return 1 - np.exp(a_par1 * x[1] + b_par1) * (x[0] * f_d) ** b1_par1 - np.exp(c_par1 * x[1] + d_par1) * (
                x[0] * f_d) ** (e_par1 * x[1] + f_par1)
        
        if self.aTextEdit_eu.text() == "":
            parini1 = [0.03, -18, 0.7, 2.3, -782, -0.28, 96, 1]
            self.aTextEdit_eu.setText(str(parini1[0]))
            self.bTextEdit_eu.setText(str(parini1[1]))
            self.b1TextEdit_eu.setText(str(parini1[2]))
            self.cTextEdit_eu.setText(str(parini1[3]))
            self.dTextEdit_eu.setText(str(parini1[4]))
            self.eTextEdit_eu.setText(str(parini1[5]))
            self.fTextEdit_eu.setText(str(parini1[6]))
            self.fdTextEdit_eu.setText(str(parini1[7]))
        
        # UI 변수를 초기치로 산정
        a_par1 = float(self.aTextEdit_eu.text())
        b_par1 = float(self.bTextEdit_eu.text())
        b1_par1 = float(self.b1TextEdit_eu.text())
        c_par1 = float(self.cTextEdit_eu.text())
        d_par1 = float(self.dTextEdit_eu.text())
        e_par1 = float(self.eTextEdit_eu.text())
        f_par1 = float(self.fTextEdit_eu.text())
        fd_par1 = float(self.fdTextEdit_eu.text())
        self.xscale = int(self.simul_x_max_eu.text())
        y_max = float(self.simul_y_max_eu.text())
        y_min = float(self.simul_y_min_eu.text())
        # parameter 계산 및 fd 산정
        root = Tk()
        root.withdraw()
        datafilepath = filedialog.askopenfilenames(initialdir="d://", title="Choose Test files")
        if datafilepath:
            self.cycparameter_eu.setText(str(datafilepath[0]))
        file = 0
        num = 0
        writerowno = 0
        eol_cycle = 0
        result_all = pd.DataFrame()
        raw_all = pd.DataFrame()
        if self.saveok.isChecked():
            save_file_name = filedialog.asksaveasfilename(initialdir="D://", title="Save File Name", defaultextension=".xlsx")
            if save_file_name:
                writer = pd.ExcelWriter(save_file_name, engine="xlsxwriter")
        # 3열 데이터 연결, 절대온도 변환, max 1로 변경
        for filepath in datafilepath:
            # tab 그래프 추가
            fig, ax = plt.subplots(figsize=(6, 6))
            tab = QtWidgets.QWidget()
            tab_layout = QtWidgets.QVBoxLayout(tab)
            canvas = FigureCanvas(fig)
            toolbar = NavigationToolbar(canvas, None)
            dfall = pd.DataFrame({'x': [], 't': [], 'y': []})
            df = pd.read_csv(filepath, sep="\t", engine="c", encoding="UTF-8", skiprows=1, on_bad_lines='skip')
            if not "02C" in df.columns:
                const_namelist = filepath.split("/")
                filemax = len(datafilepath)
                file = file + 1
                first_column_name = df.columns[1]
                if isinstance(first_column_name, str) and first_column_name.isdigit():
                    for i in range(1, len(df.columns)):
                        y1 = df.iloc[:,i]
                        x1 = df.iloc[:len(y1),0]
                        # 초기값으로 100분율 환산
                        dataadd = pd.DataFrame({'x': x1, 'y': y1})
                        dataadd = dataadd.dropna()
                        if df.iloc[0, i] != 0 and len(y1)!= 0:
                            x_0 = dataadd.iloc[0,0]
                            x_1 = dataadd.iloc[1,0]
                            y_0 = dataadd.iloc[0,1]
                            y_1 = dataadd.iloc[1,1]
                            coefficients = np.polyfit([x_0, x_1], [y_0, y_1], 1)
                            polynomial = np.poly1d(coefficients)
                            y_max_cap = polynomial(1)
                            dataadd.y = dataadd.y/y_max_cap
                        raw_all = pd.concat([dataadd, raw_all], axis=1)
                        dataadd['t'] = float(df.columns[i][:2])
                        if dataadd.t.max() < 273:
                            dataadd.t = dataadd.t + 273
                        progressdata = (file + 1 + num/len(list(df)))/filemax * 100
                        dfall = dfall._append(dataadd)
                        self.progressBar.setValue(int(progressdata))
                        if dataadd.t.max() == 296:
                            dfall = dfall[dfall.t == 296]
                            dfall = dfall.dropna()
                            maxfevset = 5000
                            p0 = [fd_par1]
                            df2 = pd.DataFrame({'x': range(1, self.xscale, 1)})
                            df2.t1 = 23 + 273
                            popt, pcov = curve_fit(cyccapparameter, (dfall.x, dfall.t), dfall.y, p0, maxfev=maxfevset)
                            residuals = dfall.y - cyccapparameter((dfall.x, dfall.t), *popt)
                            ss_res = np.sum(residuals ** 2)
                            ss_tot = np.sum((dfall.y - np.mean(dfall.y)) ** 2)
                            r_squared = 1 - (ss_res/ss_tot)
                            result1 = cyccapparameter((df2.x, df2.t1), *popt)
                            eol_cycle = (result1 - 0.8).abs().idxmin()
                            r_square_label = 'R^2 = %5.4f' % r_squared
                            acc_const = round(popt[0]/p0[0], 3)
                            self.fdTextEdit_eu.setText(str(round(popt[0], 50)))
                            filename = filepath.split(".t")[-2].split("/")[-1].split("\\")[-1]
                            ax.plot(df2.x, result1, label=r_square_label + ' / ' + str(eol_cycle) + ' / ' + str(acc_const))
                            ax.plot(dfall.x, dfall.y, 'o', color='blue', markersize=3)
                            plt.suptitle(filename, fontsize= 14, fontweight='bold')
                            graph_eu_set(ax, y_min, y_max)
                            plt.tight_layout(pad=1, w_pad=1, h_pad=1)
                            # 데이터 저장
                            result0 = df2.x
                            result = pd.DataFrame({"x": result0, str(df2.t1 - 273): result1})
                            result_all = pd.concat([result_all, result], axis=1)
                            result_para = pd.DataFrame({"para": popt})
                            if self.saveok.isChecked() and save_file_name:
                                output_data(result_para, "parameter", writerowno + 1, 1, "para", [const_namelist[-1]])
                                writerowno = writerowno + 1
                            tab_layout.addWidget(toolbar)
                            tab_layout.addWidget(canvas)
                            self.cycle_simul_tab_eu.addTab(tab, filename)
                            self.cycle_simul_tab_eu.setCurrentWidget(tab)
                            plt.tight_layout(pad=1, w_pad=1, h_pad=1)
                else:
                    for i in range(0, len(df.columns)):
                        num = num + 1
                        if i % 3 == 2:
                            y1 = df.iloc[:, i]
                            x1 = df.iloc[:, i - 2][:len(y1)]
                            # 초기값으로 100분율 환산
                            if df.iloc[0, i] != 0:
                                x_0 = x1[0]
                                x_1 = x1[1]
                                y_0 = y1[0]
                                y_1 = y1[1]
                                coefficients = np.polyfit([x_0, x_1], [y_0, y_1], 1)
                                polynomial = np.poly1d(coefficients)
                                y_max_cap = polynomial(1)
                                y1 = y1/y_max_cap
                            t1 = df.iloc[:, i - 1][:len(y1)]
                            if t1.max() < 273:
                                t1 = t1 + 273
                            dfall = pd.DataFrame({'x': x1, 't': t1, 'y': y1})
                            progressdata = (file + 1 + num/len(list(df)))/filemax * 100
                            raw_all = pd.concat([dfall, raw_all], axis=1)
                            self.progressBar.setValue(int(progressdata))
                            if t1.max() == 296:
                                dfall = dfall[dfall.t == 296]
                                dfall = dfall.dropna()
                                maxfevset = 5000
                                p0 = [fd_par1]
                                df2 = pd.DataFrame({'x': range(1, self.xscale, 1)})
                                df2.t1 = 23 + 273
                                popt, pcov = curve_fit(cyccapparameter, (dfall.x, dfall.t), dfall.y, p0, maxfev=maxfevset)
                                residuals = dfall.y - cyccapparameter((dfall.x, dfall.t), *popt)
                                ss_res = np.sum(residuals ** 2)
                                ss_tot = np.sum((dfall.y - np.mean(dfall.y)) ** 2)
                                r_squared = 1 - (ss_res/ss_tot)
                                result1 = cyccapparameter((df2.x, df2.t1), *popt)
                                eol_cycle = min(eol_cycle, (result1 - 0.8).abs().idxmin())
                                r_square_label = 'R^2 = %5.4f' % r_squared
                                acc_const = round(popt[0]/p0[0], 3)
                                self.fdTextEdit_eu_2.setText(str(round(popt[0], 50)))
                                filename = filepath.split(".t")[-2].split("/")[-1].split("\\")[-1]
                                ax.plot(df2.x, result1, label=r_square_label + ' / ' + str(eol_cycle) + ' / ' + str(acc_const))
                                ax.plot(dfall.x, dfall.y, 'o', color='blue', markersize=3)
                                graph_eu_set(ax, y_min, y_max)
                                plt.suptitle(filename, fontsize= 14, fontweight='bold')
                                plt.tight_layout(pad=1, w_pad=1, h_pad=1)
                                # 데이터 저장
                                result0 = df2.x
                                result = pd.DataFrame({"x": result0, str(df2.t1 - 273): result1})
                                result_all = pd.concat([result_all, result], axis=1)
                                result_para = pd.DataFrame({"para": popt})
                                if self.saveok.isChecked() and save_file_name:
                                    output_data(result_para, "parameter", writerowno + 1, 1, "para", [const_namelist[-1]])
                                    writerowno = writerowno + 1
                                tab_layout.addWidget(toolbar)
                                tab_layout.addWidget(canvas)
                                self.cycle_simul_tab_eu.addTab(tab, filename)
                                self.cycle_simul_tab_eu.setCurrentWidget(tab)
                                plt.tight_layout(pad=1, w_pad=1, h_pad=1)
        if self.saveok.isChecked() and save_file_name:
            result_all.to_excel(writer, sheet_name="estimation", index=False)
            raw_all.to_excel(writer, sheet_name="raw", index=False)
            writer.close()
        plt.tight_layout(pad=1, w_pad=1, h_pad=1)
        self.progressBar.setValue(100)
        plt.close()

    def simulation_tab_reset_confirm_button(self):
        self.tab_delete(self.real_cycle_simul_tab)
        self.tab_no = 0
    
    def simulation_confirm_button(self):
        def BaseEquation(a_par, b_par, fd, b1_par, c_par, d_par, e_par, f_par, temp_par, so_par, x):
            return np.exp(a_par * temp_par + b_par) * (x * fd) ** b1_par + np.exp( c_par * temp_par + d_par) * (x * fd) ** (
                e_par * temp_par + f_par) - so_par
        # exp-열화모드, parameter 고정 후 열화 parameter 계산 (각각에 soh와 soir을 빼서 수식을 0에 맞춤)
        def cyccapparameter(x):
            return (1 - BaseEquation(a_par1, b_par1, cycle_cap_simul_fd, b1_par1, c_par1, d_par1, e_par1, f_par1, cycle_temp, (-1) * soh, x))
        def cycirparameter(x):
            return BaseEquation(a_par3, b_par3, cycle_dcir_simul_fd, b1_par3, c_par3, d_par3, e_par3, f_par3, cycle_temp, soir, x)
        # parameter 고정 후 열화 parameter 계산
        def stgcapparameter(x):
            return (1 - BaseEquation(a_par2, b_par2, storage2_cap_simul_fd, b1_par2, c_par2, d_par2, e_par2, f_par2, storage_temp2, (-1) * soh, x))
        def stgirparameter(x):
            return BaseEquation(a_par4, b_par4, storage2_dcir_simul_fd, b1_par4, c_par4, d_par4, e_par4, f_par4, storage_temp2, soir, x)
        def stgcapparameter2(x):
            return (1 - BaseEquation(a_par2, b_par2, storage1_cap_simul_fd, b1_par2, c_par2, d_par2, e_par2, f_par2, storage_temp1, (-1) * soh, x))
        def stgirparameter2(x):
            return BaseEquation(a_par4, b_par4, storage1_dcir_simul_fd, b1_par4, c_par4, d_par4, e_par4, f_par4, storage_temp1, soir, x)
        # parameter 계산 및 fd 산정
        root = Tk()
        root.withdraw()
        # parameter folder 지정
        dirname = filedialog.askdirectory(initialdir="d://parameter", title="Choose Test Folders") 
        self.capparameterload_path.setText(dirname)
        # 용량 parameter 설정
        if os.path.isfile(dirname + "//para_cyccapparameter.txt"):
            df_cyc = pd.read_csv(dirname + "//para_cyccapparameter.txt", sep="\t", engine="c", encoding="UTF-8", skiprows=1, on_bad_lines='skip')
        if os.path.isfile(dirname + "/para_stgcapparameter.txt"):
            df_stg = pd.read_csv(dirname + "//para_stgcapparameter.txt", sep="\t", engine="c", encoding="UTF-8", skiprows=1, on_bad_lines='skip')
        if os.path.isfile(dirname + "//para_capparameter.txt"):
            df_par = pd.read_csv(dirname + "//para_capparameter.txt", sep="\t", engine="c", encoding="UTF-8", skiprows=1, on_bad_lines='skip')
        # 저항 parameter 설정
        if os.path.isfile(dirname + "//para_cycirparameter.txt"):
            df_cyc2 = pd.read_csv(dirname + "//para_cycirparameter.txt", sep="\t", engine="c", encoding="UTF-8", skiprows=1, on_bad_lines='skip')
        if os.path.isfile(dirname + "//para_stgirparameter.txt"):
            df_stg2 = pd.read_csv(dirname + "//para_stgirparameter.txt", sep="\t", engine="c", encoding="UTF-8", skiprows=1, on_bad_lines='skip')
        if os.path.isfile(dirname + "//para_irparameter.txt"):
            df_par2 = pd.read_csv(dirname + "//para_irparameter.txt", sep="\t", engine="c", encoding="UTF-8", skiprows=1, on_bad_lines='skip')
        self.aTextEdit.setText(str(round(df_par.iloc[0, 0], 50)))
        self.bTextEdit.setText(str(round(df_par.iloc[1, 0], 50)))
        self.b1TextEdit.setText(str(round(df_par.iloc[2, 0], 50)))
        self.cTextEdit.setText(str(round(df_par.iloc[3, 0], 50)))
        self.dTextEdit.setText(str(round(df_par.iloc[4, 0], 50)))
        self.eTextEdit.setText(str(round(df_par.iloc[5, 0], 50)))
        self.fTextEdit.setText(str(round(df_par.iloc[6, 0], 50)))
        self.fdTextEdit.setText(str(round(df_par.iloc[7, 0], 50)))
        self.aTextEdit_2.setText(str(round(df_par.iloc[0, 1], 50)))
        self.bTextEdit_2.setText(str(round(df_par.iloc[1, 1], 50)))
        self.b1TextEdit_2.setText(str(round(df_par.iloc[2, 1], 50)))
        self.cTextEdit_2.setText(str(round(df_par.iloc[3, 1], 50)))
        self.dTextEdit_2.setText(str(round(df_par.iloc[4, 1], 50)))
        self.eTextEdit_2.setText(str(round(df_par.iloc[5, 1], 50)))
        self.fTextEdit_2.setText(str(round(df_par.iloc[6, 1], 50)))
        self.fdTextEdit_2.setText(str(round(df_par.iloc[7, 1], 50)))
        self.aTextEdit_3.setText(str(round(df_par2.iloc[0, 0], 50)))
        self.bTextEdit_3.setText(str(round(df_par2.iloc[1, 0], 50)))
        self.b1TextEdit_3.setText(str(round(df_par2.iloc[2, 0], 50)))
        self.cTextEdit_3.setText(str(round(df_par2.iloc[3, 0], 50)))
        self.dTextEdit_3.setText(str(round(df_par2.iloc[4, 0], 50)))
        self.eTextEdit_3.setText(str(round(df_par2.iloc[5, 0], 50)))
        self.fTextEdit_3.setText(str(round(df_par2.iloc[6, 0], 50)))
        self.fdTextEdit_3.setText(str(round(df_par2.iloc[7, 0], 50)))
        self.aTextEdit_4.setText(str(round(df_par2.iloc[0, 1], 50)))
        self.bTextEdit_4.setText(str(round(df_par2.iloc[1, 1], 50)))
        self.b1TextEdit_4.setText(str(round(df_par2.iloc[2, 1], 50)))
        self.cTextEdit_4.setText(str(round(df_par2.iloc[3, 1], 50)))
        self.dTextEdit_4.setText(str(round(df_par2.iloc[4, 1], 50)))
        self.eTextEdit_4.setText(str(round(df_par2.iloc[5, 1], 50)))
        self.fTextEdit_4.setText(str(round(df_par2.iloc[6, 1], 50)))
        self.fdTextEdit_4.setText(str(round(df_par2.iloc[7, 1], 50)))
        a_par1 = float(self.aTextEdit.text())
        b_par1 = float(self.bTextEdit.text())
        b1_par1 = float(self.b1TextEdit.text())
        c_par1 = float(self.cTextEdit.text())
        d_par1 = float(self.dTextEdit.text())
        e_par1 = float(self.eTextEdit.text())
        f_par1 = float(self.fTextEdit.text())
        a_par2 = float(self.aTextEdit_2.text())
        b_par2 = float(self.bTextEdit_2.text())
        b1_par2 = float(self.b1TextEdit_2.text())
        c_par2 = float(self.cTextEdit_2.text())
        d_par2 = float(self.dTextEdit_2.text())
        e_par2 = float(self.eTextEdit_2.text())
        f_par2 = float(self.fTextEdit_2.text())
        a_par3 = float(self.aTextEdit_3.text())
        b_par3 = float(self.bTextEdit_3.text())
        b1_par3 = float(self.b1TextEdit_3.text())
        c_par3 = float(self.cTextEdit_3.text())
        d_par3 = float(self.dTextEdit_3.text())
        e_par3 = float(self.eTextEdit_3.text())
        f_par3 = float(self.fTextEdit_3.text())
        a_par4 = float(self.aTextEdit_4.text())
        b_par4 = float(self.bTextEdit_4.text())
        b1_par4 = float(self.b1TextEdit_4.text())
        c_par4 = float(self.cTextEdit_4.text())
        d_par4 = float(self.dTextEdit_4.text())
        e_par4 = float(self.eTextEdit_4.text())
        f_par4 = float(self.fTextEdit_4.text())
        # 입력 condition
        all_input_data_path = filedialog.askopenfilenames(initialdir = dirname, title="Choose Test files")
        # while self.real_cycle_simul_tab.count() > 0:
        #     self.real_cycle_simul_tab.removeTab(0)
        for input_data_path in all_input_data_path:
            fig, ((axe1, axe4), (axe2, axe5)) = plt.subplots(nrows=2, ncols=2, figsize=(12, 9))
            tab = QtWidgets.QWidget()
            tab_layout = QtWidgets.QVBoxLayout(tab)
            canvas = FigureCanvas(fig)
            toolbar = NavigationToolbar(canvas, None)
            if input_data_path != "":
                input_data = pd.read_csv(input_data_path, sep="\t", engine="c", encoding="UTF-8", skiprows=1, on_bad_lines='skip')
                # 장수명
                self.txt_longcycleno.setText(str(input_data.iloc[0, 1]))
                self.txt_longcyclevol.setText(str(input_data.iloc[1, 1]))
                self.txt_relcap.setText(str(input_data.iloc[2, 1]))
                # 사이클
                self.xaxixTextEdit.setText(str(int(input_data.iloc[3, 1])))
                self.UsedCapTextEdit.setText(str(input_data.iloc[4, 1]))
                self.DODTextEdit.setText(str(input_data.iloc[5, 1]))
                # 충전 조건
                self.CrateTextEdit.setText(str(input_data.iloc[6, 1]))
                self.SOCTextEdit.setText(str(input_data.iloc[7, 1]))
                self.DcrateTextEdit.setText(str(input_data.iloc[8, 1]))
                self.TempTextEdit.setText(str(input_data.iloc[9, 1]))
                # 저장
                self.SOCTextEdit_3.setText(str(input_data.iloc[10, 1]))
                self.TempTextEdit_3.setText(str(input_data.iloc[11, 1]))
                self.RestTextEdit_2.setText(str(input_data.iloc[12, 1]))
                # 저장2
                self.SOCTextEdit_2.setText(str(input_data.iloc[13, 1]))
                self.TempTextEdit_2.setText(str(input_data.iloc[14, 1]))
                self.RestTextEdit.setText(str(input_data.iloc[15, 1]))
            # SaveFileName set-up
            if self.saveok.isChecked():
                save_file_name = filedialog.asksaveasfilename(initialdir="D://", title="Save File Name", defaultextension=".xlsx")
            # cycle 인자
            self.xscale = int(self.xaxixTextEdit.text())
            cycle_crate = float(self.CrateTextEdit.text())
            cycle_dcrate = float(self.DcrateTextEdit.text())
            cycle_soc = float(self.SOCTextEdit.text())
            cycle_dod = float(self.DODTextEdit.text())
            cycle_temp = float(self.TempTextEdit.text()) + 273
            # 저장1 인자
            storage_temp1 = float(self.TempTextEdit_3.text()) + 273
            storage_rest1 = float(self.RestTextEdit_2.text())
            storage_soc1 = float(self.SOCTextEdit_3.text())
            # 저장2 인자
            storage_temp2 = float(self.TempTextEdit_2.text()) + 273
            storage_rest2 = float(self.RestTextEdit.text())
            storage_soc2 = float(self.SOCTextEdit_2.text())
            usedcap = float(self.UsedCapTextEdit.text())
            cycle, time, soh, soir = 0, 0, 1, 0
            degree_storage1_cap = 0
            degree_storage1_dcir = 0
            degree_storage2_cap = 0
            degree_storage2_dcir = 0
            storagecycle = 0
            stgcountratio = float(self.txt_storageratio.text())
            stgcountratio2 = float(self.txt_storageratio2.text())
            result = pd.DataFrame(
                {"cycle": 0,
                "time": 0,
                "storagecycle": 0,
                "degree_cycle_cap": 0,
                "degree_storage1_cap": 0,
                "degree_storage2_cap": 0,
                "degree_cycle_dcir": 0,
                "degree_storage1_dcir": 0,
                "degree_storage2_dcir": 0,
                "SOH": 1,
                "rSOH": 1,
                "SOIR": 0},
                index=[0])
            for i in range(0, 100000):
                if self.nolonglife.isChecked():
                    # 장수명 미적용
                    cycle_soc_cal = cycle_soc
                    storage_soc1_cal = storage_soc1
                    storage_soc2_cal = storage_soc2
                    long_cycle = list(map(int, (self.txt_longcycleno.text().split())))
                    long_cycle_vol = list(map(float, (self.txt_longcyclevol.text().split())))
                    real_cap = list(map(float, (self.txt_relcap.text().split())))
                    cycle_soc_cal = cycle_soc - long_cycle_vol[0]
                    storage_soc1_cal = storage_soc1 - long_cycle_vol[0]
                    storage_soc2_cal = storage_soc2 - long_cycle_vol[0]
                    para_rsoh = real_cap[0]/100
                elif self.hhp_longlife.isChecked():
                    # 장수명 적용 parameter
                    long_cycle = list(map(int, (self.txt_longcycleno.text().split())))
                    long_cycle_vol = list(map(float, (self.txt_longcyclevol.text().split())))
                    real_cap = list(map(float, (self.txt_relcap.text().split())))
                    # 장수명 적용 시 전압 변화, 실용량 변화 산정
                    # compare to long cycle and storge cycle
                    if storagecycle > cycle:
                        complexcycle = storagecycle 
                    else:
                        complexcycle = cycle
                    # 장수명 마지막 스텝
                    cycle_soc_cal = cycle_soc - long_cycle_vol[len(long_cycle) - 1]
                    storage_soc1_cal = storage_soc1 - long_cycle_vol[len(long_cycle) - 1]
                    storage_soc2_cal = storage_soc2 - long_cycle_vol[len(long_cycle) - 1]
                    para_rsoh = real_cap[len(long_cycle) - 1] / 100
                    # 사이클에 따른 장수명 계산
                    for cyc_i in range(0, len(long_cycle) - 1):
                        if (complexcycle >= long_cycle[cyc_i]) and (complexcycle <= long_cycle[cyc_i + 1]):
                            cycle_soc_cal = cycle_soc - long_cycle_vol[cyc_i]
                            storage_soc1_cal = storage_soc1 - long_cycle_vol[cyc_i]
                            storage_soc2_cal = storage_soc2 - long_cycle_vol[cyc_i]
                            para_rsoh = real_cap[cyc_i]/100
                # 사이클 Fd 구하기 (타펠식, 아레니우스식의 복합식 사용) - 용량/저항, 기준 fd를 base로 가속비를 곱해서 계산
                if "df_cyc" in locals():
                    cycle_cap_simul_fd = (df_cyc.Crate[0] * cycle_crate + df_cyc.Crate[1]) * (df_cyc.SOC[0] * cycle_soc_cal + df_cyc.SOC[1]) * (
                        df_cyc.DOD[0] * cycle_dod + df_cyc.DOD[1]) * (df_cyc.fd[0])
                    cycle_dcir_simul_fd = (df_cyc2.Crate[0] * cycle_crate + df_cyc2.Crate[1]) * (df_cyc2.SOC[0] * cycle_soc_cal + df_cyc2.SOC[1]) * (
                        df_cyc2.DOD[0] * cycle_dod + df_cyc2.DOD[1]) * (df_cyc2.fd[0])
                else:
                    err_msg('파일 or 경로없음!!','Check condition file !!')
                    return
                if "df_stg" in locals():
                    # 저장1 Fd 구하기 (Tafel식 사용 delta E = a ln i) - 용량/저항, 직접 fd를 산출
                    storage1_cap_simul_fd = np.exp(df_stg.iloc[0, 0] * storage_soc1_cal + df_stg.iloc[1, 0])
                    storage1_dcir_simul_fd = np.exp(df_stg2.iloc[0, 0] * storage_soc1_cal + df_stg2.iloc[1, 0])
                    # 저장2 Fd 구하기 (Tafel식 사용 delta E = a ln i) - 용량/저항, 직접 fd를 산출
                    storage2_cap_simul_fd = np.exp(df_stg.iloc[0, 0] * storage_soc2_cal + df_stg.iloc[1, 0])
                    storage2_dcir_simul_fd = np.exp(df_stg2.iloc[0, 0] * storage_soc2_cal + df_stg2.iloc[1, 0])
                else:
                    err_msg('파일 or 경로없음!!','Check condition file !!')
                    return
                # soh 기준으로 cycle 환산 - SET 기준
                cycle = cycle + usedcap/para_rsoh
                # 충전, 방전 기준으로 시간 산정
                time = usedcap/cycle_crate/24 + usedcap/cycle_dcrate/24 + time
                # 수명-역으로 값을 찾는 수식
                inverse_cycle_cap_soh = root_scalar(cyccapparameter, bracket=[0, 500000], method='brentq')
                inverse_cycle_dcir_soh = root_scalar(cycirparameter, bracket=[0, 500000], method='brentq')
                solve_inverse_cycle_cap_soh = inverse_cycle_cap_soh.root
                solve_inverse_cycle_dcir_soh = inverse_cycle_dcir_soh.root
                if np.isnan(solve_inverse_cycle_cap_soh):
                    solve_inverse_cycle_cap_soh = 0
                    solve_inverse_cycle_dcir_soh = 0
                # 수명 열화 - 사용 용량으로 열화 계산
                degree_cycle_cap = (cyccapparameter(solve_inverse_cycle_cap_soh) - cyccapparameter(solve_inverse_cycle_cap_soh + usedcap))
                degree_cycle_dcir = (cycirparameter(solve_inverse_cycle_dcir_soh + usedcap) - cycirparameter(solve_inverse_cycle_dcir_soh))
                soh = soh - degree_cycle_cap
                rsoh = soh * para_rsoh
                soir = soir + degree_cycle_dcir
                cycle_result = pd.DataFrame(
                    {"cycle": cycle,
                    "time": time,
                    "storagecycle": storagecycle,
                    "degree_cycle_cap": degree_cycle_cap,
                    "degree_storage1_cap": degree_storage1_cap,
                    "degree_storage2_cap": degree_storage2_cap,
                    "degree_cycle_dcir": degree_cycle_dcir,
                    "degree_storage1_dcir": degree_storage1_dcir,
                    "degree_storage2_dcir": degree_storage2_dcir,
                    "SOH": soh,
                    "rSOH": rsoh,
                    "SOIR": soir},
                    index=[3 * i + 1])
                result = pd.concat([result, cycle_result])
                # 저장 1차-역으로 값을 찾는 수식
                inverse_storage1_cap_soh = root_scalar(stgcapparameter2, bracket=[0, 500000], method='brentq')
                inverse_storage1_dcir_soh = root_scalar(stgirparameter2, bracket=[0, 500000], method='brentq')
                solve_inverse_storage1_cap_soh = inverse_storage1_cap_soh.root
                solve_inverse_storage1_dcir_soh = inverse_storage1_dcir_soh.root
                if np.isnan(solve_inverse_storage1_cap_soh):
                    solve_inverse_storage1_cap_soh = 0
                    solve_inverse_storage1_dcir_soh = 0
                # 저장 1차 열화
                time = time + storage_rest1
                if stgcountratio == 0:
                    storagecycle = storagecycle
                else:
                    storagecycle = storagecycle + storage_rest1/(stgcountratio/24)
                degree_storage1_cap = (stgcapparameter2(solve_inverse_storage1_cap_soh) - stgcapparameter2(
                    solve_inverse_storage1_cap_soh + storage_rest1))
                degree_storage1_dcir = (stgirparameter2(solve_inverse_storage1_dcir_soh + storage_rest1) - stgirparameter2(
                    solve_inverse_storage1_dcir_soh))
                soh = soh - degree_storage1_cap
                rsoh = soh * para_rsoh
                soir = soir + degree_storage1_dcir
                storage1_result = pd.DataFrame(
                    {"cycle": cycle,
                    "time": time,
                    "storagecycle": storagecycle,
                    "degree_cycle_cap": degree_cycle_cap,
                    "degree_storage1_cap": degree_storage1_cap,
                    "degree_storage2_cap": degree_storage2_cap,
                    "degree_cycle_dcir": degree_cycle_dcir,
                    "degree_storage1_dcir": degree_storage1_dcir,
                    "degree_storage2_dcir": degree_storage2_dcir,
                    "SOH": soh,
                    "rSOH": rsoh,
                    "SOIR": soir},
                    index=[3 * i + 2])
                result = pd.concat([result, storage1_result])
                # 저장 2차-역으로 값을 찾는 수식
                inverse_storage2_cap_soh = root_scalar(stgcapparameter, bracket=[0, 500000], method='brentq')
                inverse_storage2_dcir_soh = root_scalar(stgirparameter, bracket=[0, 500000], method='brentq')
                solve_inverse_storage2_cap_soh = inverse_storage2_cap_soh.root
                solve_inverse_storage2_dcir_soh = inverse_storage2_dcir_soh.root
                if np.isnan(solve_inverse_storage2_cap_soh):
                    solve_inverse_storage2_cap_soh = 0
                    solve_inverse_storage2_dcir_soh = 0
                # 저장 2차 열화
                time = time + storage_rest2
                if stgcountratio2 == 0:
                    storagecycle = storagecycle
                else:
                    storagecycle = storagecycle + storage_rest2/(stgcountratio2/24)
                degree_storage2_cap = (stgcapparameter(solve_inverse_storage2_cap_soh) - stgcapparameter(
                    solve_inverse_storage2_cap_soh + storage_rest2))
                degree_storage2_dcir = (stgirparameter(solve_inverse_storage2_dcir_soh + storage_rest2) - stgirparameter(
                    solve_inverse_storage2_dcir_soh))
                soh = soh - degree_storage2_cap
                rsoh = soh * para_rsoh
                soir = soir + degree_storage2_dcir
                storage2_result = pd.DataFrame(
                    {"cycle": cycle,
                    "time": time,
                    "storagecycle": storagecycle,
                    "degree_cycle_cap": degree_cycle_cap,
                    "degree_storage1_cap": degree_storage1_cap,
                    "degree_storage2_cap": degree_storage2_cap,
                    "degree_cycle_dcir": degree_cycle_dcir,
                    "degree_storage1_dcir": degree_storage1_dcir,
                    "degree_storage2_dcir": degree_storage2_dcir,
                    "SOH": soh,
                    "rSOH": rsoh,
                    "SOIR": soir},
                    index=[3 * i + 3])
                result = pd.concat([result, storage2_result])
                if soh < 0.75 or np.isnan(soh) or cycle > self.xscale:
                    break
                self.progressBar.setValue(int((1 - soh)/0.5 * 100))
            
            # tab 그래프 추가
            result.cycdeg_sum = 1 - result.degree_cycle_cap.cumsum()/3
            result.cycir_sum = result.degree_cycle_dcir.cumsum()/3
            result.stgdeg_sum1 = 1 - result.degree_storage1_cap.cumsum()/3
            result.stgir_sum1 = result.degree_storage1_dcir.cumsum()/3
            result.stgdeg_sum2 = 1 - result.degree_storage2_cap.cumsum()/3
            result.stgir_sum2 = result.degree_storage2_dcir.cumsum()/3
            # 전압에 의한 가속 계수 선정
            self.FdTextEdit.setText(str(round(cycle_cap_simul_fd, 50)))
            self.FdTextEdit_3.setText(str(round(cycle_dcir_simul_fd, 50)))
            self.FdTextEdit_2.setText(str(round(storage2_cap_simul_fd, 50)))
            self.FdTextEdit_4.setText(str(round(storage2_dcir_simul_fd, 50)))
            self.FdTextEdit_5.setText(str(round(storage1_cap_simul_fd, 50)))
            self.FdTextEdit_6.setText(str(round(storage1_dcir_simul_fd, 50)))
            if self.chk_cell_cycle.isChecked():
                graph_simulation(axe1, result.cycle, result.SOH, 'b-', 'Cell Capacity', self.xscale, 0.75, 1, 'cycle', 'Capacity')
            if self.chk_set_cycle.isChecked():
                graph_simulation(axe1, result.cycle, result.rSOH, 'k-', 'SET Capacity', self.xscale, 0.75, 1, 'cycle', 'Capacity')
            if self.chk_detail_cycle.isChecked():
                graph_simulation(axe1, result.cycle, result.cycdeg_sum, 'r-', 'soh_cyc', self.xscale, 0.75, 1, 'cycle', 'Capacity')
                graph_simulation(axe1, result.cycle, result.stgdeg_sum1, 'g-', 'soh_stg1', self.xscale, 0.75, 1, 'cycle', 'Capacity')
                graph_simulation(axe1, result.cycle, result.stgdeg_sum2, 'm-', 'soh_stg2', self.xscale, 0.75, 1, 'cycle', 'Capacity')
            if self.chk_cell_cycle.isChecked():
                graph_simulation(axe4, result.time, result.SOH, 'b-', 'Cell Capacity', self.xscale, 0.75, 1, 'day', 'Capacity')
            if self.chk_set_cycle.isChecked():
                graph_simulation(axe4, result.time, result.rSOH, 'k-', 'SET Capacity', self.xscale, 0.75, 1, 'day', 'Capacity')
            if self.chk_detail_cycle.isChecked():
                graph_simulation(axe4, result.time, result.cycdeg_sum, 'r-', 'soh_cyc', self.xscale, 0.75, 1, 'day', 'Capacity')
                graph_simulation(axe4, result.time, result.stgdeg_sum1, 'g-', 'soh_stg1', self.xscale, 0.75, 1, 'day', 'Capacity')
                graph_simulation(axe4, result.time, result.stgdeg_sum2, 'm-', 'soh_stg2', self.xscale, 0.75, 1, 'day', 'Capacity')
            graph_simulation(axe2, result.cycle, result.SOIR, 'b-', 'Swelling', self.xscale, 0, 0.08, 'cycle', 'Swelling')
            if self.chk_detail_cycle.isChecked():
                graph_simulation(axe2, result.cycle, result.cycir_sum, 'r-', 'DCIR_cyc', self.xscale, 0, 0.08, 'cycle', 'Swelling')
                graph_simulation(axe2, result.cycle, result.stgir_sum1, 'g-', 'DCIR_stg1', self.xscale, 0, 0.08, 'cycle', 'Swelling')
                graph_simulation(axe2, result.cycle, result.stgir_sum2, 'm-', 'DCIR_stg2', self.xscale, 0, 0.08, 'cycle', 'Swelling')
            graph_simulation(axe5, result.time, result.SOIR, 'b-', 'Swelling', self.xscale, 0, 0.08, 'day', 'Swelling')
            if self.chk_detail_cycle.isChecked():
                graph_simulation(axe5, result.time, result.cycir_sum, 'r-', 'DCIR_cyc', self.xscale, 0, 0.08, 'day', 'Swelling')
                graph_simulation(axe5, result.time, result.stgir_sum1, 'g-', 'DCIR_stg1', self.xscale, 0, 0.08, 'day', 'Swelling')
                graph_simulation(axe5, result.time, result.stgir_sum2, 'm-', 'DCIR_stg2', self.xscale, 0, 0.08, 'day', 'Swelling')
            plt.tight_layout(pad=1, w_pad=1, h_pad=1)
            if input_data_path != "":
                filename = input_data_path.split(".t")[-2].split("/")[-1].split("\\")[-1]
            else:
                filename = "simulation"
            tab_layout.addWidget(toolbar)
            tab_layout.addWidget(canvas)
            self.real_cycle_simul_tab.addTab(tab, filename)
            self.real_cycle_simul_tab.setCurrentWidget(tab)
            plt.tight_layout(pad=1, w_pad=1, h_pad=1)
            if self.saveok.isChecked():
                result.to_excel("simul" + filename, index=False)
                output_fig(self.figsaveok, "fig" + filename)
        plt.tight_layout(pad=1, w_pad=1, h_pad=1)
        self.progressBar.setValue(100)
            # plt.show()
        plt.close()

    def ptn_change_pattern_button(self):
        # ui에서 데이터 확인
        self.progressBar.setValue(0)
        ptn_ori_path = str(self.ptn_ori_path.text())
        ptn_crate = float(self.ptn_crate.text())
        ptn_capacity = float(self.ptn_capacity.text())
        # 파일 있는지 확인
        if not os.path.isfile(ptn_ori_path):
            ptn_ori_path = filedialog.askopenfilename(initialdir="c:\\Program Files\\PNE CTSPro\\Database\\Cycler_Schedule_2000.mdbd",
                                                      title="Choose Test files")
            self.ptn_ori_path.setText(str(ptn_ori_path))
        conn_str = (
            r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
            r'DBQ=' + ptn_ori_path + ';')
        conn =pyodbc.connect(conn_str)
        # 쿼리 실행
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT TestID FROM Step;")
        if (not hasattr(self, "ptn_df_select")) or (len(self.ptn_df_select) == 0) or (self.ptn_df_select[0] == ""):
            pass
        else:
            for testidcount in self.ptn_df_select:
                cursor.execute("SELECT MAX(Iref) From Step WHERE TestID = ? AND StepType = 2", str(testidcount))
                max_value = cursor.fetchone()[0]
                if max_value is not None:
                    base_capacity = max_value / ptn_crate
                    real_capacity = ptn_capacity
                    if self.chk_coincell.isChecked():
                        cursor.execute("UPDATE Step SET Iref = -round(-Iref /? *?, 3) WHERE TestID =?",
                                    str(base_capacity), str(real_capacity), str(testidcount))
                        cursor.execute("UPDATE Step SET EndI = -round(-EndI /? *?, 3) WHERE TestID =?", 
                                    str(base_capacity), str(real_capacity), str(testidcount))
                    else:
                        cursor.execute("UPDATE Step SET Iref = -int(-Iref /? *?) WHERE TestID =?",
                                    str(base_capacity), str(real_capacity), str(testidcount))
                        cursor.execute("UPDATE Step SET EndI = -int(-EndI /? *?) WHERE TestID =?", 
                                    str(base_capacity), str(real_capacity), str(testidcount))
        # 변경 사항 저장
        conn.commit()
        # 커서 및 연결 닫기
        cursor.close()
        conn.close()
        self.progressBar.setValue(100)

    def ptn_change_refi_button(self):
        self.progressBar.setValue(0)
        # ui에서 데이터 확인
        ptn_ori_path = str(self.ptn_ori_path.text())
        ptn_refi_pre = float(self.ptn_refi_pre.text())
        ptn_refi_after = float(self.ptn_refi_after.text())
        # 파일 있는지 확인
        if not os.path.isfile(ptn_ori_path):
            ptn_ori_path = filedialog.askopenfilename(initialdir="c:\\Program Files\\PNE CTSPro\\Database\\Cycler_Schedule_2000.mdbd",
                                                      title="Choose Test files")
            self.ptn_ori_path.setText(str(ptn_ori_path))
        conn_str = (
            r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
            r'DBQ=' + ptn_ori_path + ';')
        conn =pyodbc.connect(conn_str)
        # 쿼리 실행
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT TestID FROM Step;")
        if (not hasattr(self, "ptn_df_select")) or (self.ptn_df_select[0] == ""):
            pass
        else:
            for testidcount in self.ptn_df_select:
                # 문자 인식 문제를 위한 강제 변환
                if self.chk_coincell.isChecked():
                    cursor.execute("UPDATE Step SET Iref = round(Iref /? *?, 3) WHERE TestID =?",
                                str(1), str(1), str(testidcount))
                    cursor.execute("UPDATE Step SET Iref = ? WHERE Iref = ? AND TestID = ?",
                                str(ptn_refi_after), str(ptn_refi_pre), str(testidcount))
                else:
                    cursor.execute("UPDATE Step SET Iref = int(Iref /? *?) WHERE TestID =?",
                                str(1), str(1), str(testidcount))
                    cursor.execute("UPDATE Step SET Iref = ? WHERE Iref = ? AND TestID = ?",
                                str(ptn_refi_after), str(ptn_refi_pre), str(testidcount))
        # 변경 사항 저장
        conn.commit()
        # 커서 및 연결 닫기
        cursor.close()
        conn.close()
        self.progressBar.setValue(100)

    def ptn_change_chgv_button(self):
        self.progressBar.setValue(0)
        # ui에서 데이터 확인
        ptn_ori_path = str(self.ptn_ori_path.text())
        ptn_chgv_pre = float(self.ptn_chgv_pre.text())
        ptn_chgv_after = float(self.ptn_chgv_after.text())
        # 파일 있는지 확인
        if not os.path.isfile(ptn_ori_path):
            ptn_ori_path = filedialog.askopenfilename(initialdir="c:\\Program Files\\PNE CTSPro\\Database\\Cycler_Schedule_2000.mdbd",
                                                      title="Choose Test files")
            self.ptn_ori_path.setText(str(ptn_ori_path))
        conn_str = (
            r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
            r'DBQ=' + ptn_ori_path + ';')
        conn =pyodbc.connect(conn_str)
        # 쿼리 실행
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT TestID FROM Step;")
        if (not hasattr(self, "ptn_df_select")) or (self.ptn_df_select[0] == ""):
            pass
        else:
            for testidcount in self.ptn_df_select:
                cursor.execute("UPDATE Step SET Vref_Charge = int(Vref_Charge /? *?) WHERE TestID =?",
                            str(1), str(1), str(testidcount))
                cursor.execute("UPDATE Step SET Vref_Charge = ? WHERE Vref_Charge = ? AND TestID =?",
                            str(ptn_chgv_after), str(ptn_chgv_pre), str(testidcount))
        # 변경 사항 저장
        conn.commit()
        # 커서 및 연결 닫기
        cursor.close()
        conn.close()
        self.progressBar.setValue(100)

    def ptn_change_dchgv_button(self):
        self.progressBar.setValue(0)
        # ui에서 데이터 확인
        ptn_ori_path = str(self.ptn_ori_path.text())
        ptn_dchgv_pre = float(self.ptn_dchgv_pre.text())
        ptn_dchgv_after = float(self.ptn_dchgv_after.text())
        # 파일 있는지 확인
        if not os.path.isfile(ptn_ori_path):
            ptn_ori_path = filedialog.askopenfilename(initialdir="c:\\Program Files\\PNE CTSPro\\Database\\Cycler_Schedule_2000.mdbd",
                                                      title="Choose Test files")
            self.ptn_ori_path.setText(str(ptn_ori_path))
        conn_str = (
            r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
            r'DBQ=' + ptn_ori_path + ';')
        conn =pyodbc.connect(conn_str)
        # 쿼리 실행
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT TestID FROM Step;")
        if (not hasattr(self, "ptn_df_select")) or (self.ptn_df_select[0] == ""):
            pass
        else:
            for testidcount in self.ptn_df_select:
                cursor.execute("UPDATE Step SET Vref_DisCharge = int(Vref_DisCharge /? *?) WHERE TestID = ?",
                            str(1), str(1), str(testidcount))
                cursor.execute("UPDATE Step SET Vref_DisCharge = ? WHERE Vref_DisCharge = ? AND TestID = ?",
                            str(ptn_dchgv_after), str(ptn_dchgv_pre), str(testidcount))
        # 변경 사항 저장
        conn.commit()
        # 커서 및 연결 닫기
        cursor.close()
        conn.close()
        self.progressBar.setValue(100)

    def ptn_change_endv_button(self):
        self.progressBar.setValue(0)
        # ui에서 데이터 확인
        ptn_ori_path = str(self.ptn_ori_path.text())
        ptn_endv_pre = float(self.ptn_endv_pre.text())
        ptn_endv_after = float(self.ptn_endv_after.text())
        # 파일 있는지 확인
        if not os.path.isfile(ptn_ori_path):
            ptn_ori_path = filedialog.askopenfilename(initialdir="c:\\Program Files\\PNE CTSPro\\Database\\Cycler_Schedule_2000.mdbd",
                                                      title="Choose Test files")
            self.ptn_ori_path.setText(str(ptn_ori_path))
        conn_str = (
            r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
            r'DBQ=' + ptn_ori_path + ';')
        conn =pyodbc.connect(conn_str)
        # 쿼리 실행
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT TestID FROM Step;")
        if (not hasattr(self, "ptn_df_select")) or (self.ptn_df_select[0] == ""):
            pass
        else:
            for testidcount in self.ptn_df_select:
                cursor.execute("UPDATE Step SET EndV = int(EndV /? *?) WHERE TestID = ?",
                            str(1), str(1), str(testidcount))
                cursor.execute("UPDATE Step SET EndV = ? WHERE EndV = ? AND TestID = ?",
                            str(ptn_endv_after), str(ptn_endv_pre), str(testidcount))
        # 변경 사항 저장
        conn.commit()
        # 커서 및 연결 닫기
        cursor.close()
        conn.close()
        self.progressBar.setValue(100)

    def ptn_change_endi_button(self):
        self.progressBar.setValue(0)
        # ui에서 데이터 확인
        ptn_ori_path = str(self.ptn_ori_path.text())
        ptn_endi_pre = float(self.ptn_endi_pre.text())
        ptn_endi_after = float(self.ptn_endi_after.text())
        # 파일 있는지 확인
        if not os.path.isfile(ptn_ori_path):
            ptn_ori_path = filedialog.askopenfilename(initialdir="c:\\Program Files\\PNE CTSPro\\Database\\Cycler_Schedule_2000.mdbd",
                                                      title="Choose Test files")
            self.ptn_ori_path.setText(str(ptn_ori_path))
        conn_str = (
            r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
            r'DBQ=' + ptn_ori_path + ';')
        conn =pyodbc.connect(conn_str)
        # 쿼리 실행
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT TestID FROM Step;")
        if (not hasattr(self, "ptn_df_select")) or (self.ptn_df_select[0] == ""):
            pass
        else:
            for testidcount in self.ptn_df_select:
                # 문자 인식 문제를 위한 강제 변환
                if self.chk_coincell.isChecked():
                    cursor.execute("UPDATE Step SET EndI = round(EndI /? *?, 3) WHERE TestID =?",
                                str(1), str(1), str(testidcount))
                    cursor.execute("UPDATE Step SET EndI = ? WHERE EndI = ? AND TestID = ?",
                                str(ptn_endi_after), str(ptn_endi_pre), str(testidcount))
                else:
                    cursor.execute("UPDATE Step SET EndI = int(EndI /? *?) WHERE TestID =?",
                                str(1), str(1), str(testidcount))
                    cursor.execute("UPDATE Step SET EndI = ? WHERE EndI = ? AND TestID = ?",
                                str(ptn_endi_after), str(ptn_endi_pre), str(testidcount))
        # 변경 사항 저장
        conn.commit()
        # 커서 및 연결 닫기
        cursor.close()
        conn.close()
        self.progressBar.setValue(100)

    def ptn_change_step_button(self):
        self.progressBar.setValue(0)
        # ui에서 데이터 확인
        ptn_ori_path = str(self.ptn_ori_path.text())
        ptn_step_pre = int(self.ptn_step_pre.text())
        ptn_step_after = int(self.ptn_step_after.text())
        # 파일 있는지 확인
        if not os.path.isfile(ptn_ori_path):
            ptn_ori_path = filedialog.askopenfilename(initialdir="c:\\Program Files\\PNE CTSPro\\Database\\Cycler_Schedule_2000.mdbd",
                                                      title="Choose Test files")
            self.ptn_ori_path.setText(str(ptn_ori_path))
        conn_str = (
            r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
            r'DBQ=' + ptn_ori_path + ';')
        conn =pyodbc.connect(conn_str)
        # 쿼리 실행 (dataframe으로 변화, 수정 후 다시 StepID에 맞춰서 변경)
        df = pd.read_sql("SELECT * FROM Step", conn)
        # 선택한 Test ID만 기준으로 dataframe에서 변경
        df["Value2"] = df["Value2"].str.replace(str(" " + str(ptn_step_pre) + " "), str(" " + str(ptn_step_pre + ptn_step_after) + " "))
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT TestID FROM Step;")
        if (not hasattr(self, "ptn_df_select")) or (self.ptn_df_select[0] == ""):
            pass
        else:
            for testidcount in self.ptn_df_select:
                for StepIDcount in df["StepID"]:
                    cursor.execute("UPDATE Step SET Value2 = ? WHERE StepID = ? AND TestID = ?",
                                str(df[df["StepID"] == StepIDcount]["Value2"].values[0]) , str(StepIDcount), str(testidcount))
        # 변경 사항 저장
        conn.commit()
        # 커서 및 연결 닫기
        cursor.close()
        conn.close()
        self.progressBar.setValue(100)

    def ptn_load_button(self):
        self.progressBar.setValue(0)
        # ui에서 데이터 확인
        ptn_ori_path = str(self.ptn_ori_path.text())
        # 파일 있는지 확인
        if not os.path.isfile(ptn_ori_path):
            ptn_ori_path = filedialog.askopenfilename(initialdir="c:\\Program Files\\PNE CTSPro\\Database\\Cycler_Schedule_2000.mdbd",
                                                      title="Choose Test files")
            self.ptn_ori_path.setText(str(ptn_ori_path))
        conn_str = (
            r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
            r'DBQ=' + ptn_ori_path + ';')
        conn =pyodbc.connect(conn_str)
        # 쿼리 실행 (Pattern 이름 테이블을 dataframe으로 변화, 수정 후 다시 StepID에 맞춰서 변경)
        pne_ptn_df = pd.read_sql("SELECT * FROM TestName", conn)
        pne_ptn_folder_name = pd.read_sql("SELECT * FROM BatteryModel", conn)
        self.pne_ptn_merged_df = pd.merge(pne_ptn_df, pne_ptn_folder_name, on='ModelID')
        self.pne_ptn_merged_df = self.pne_ptn_merged_df[["ModelName", "TestName", "Description_x", "TestID", "No", "TestNo"]]
        self.pne_ptn_merged_df = self.pne_ptn_merged_df.sort_values(by=['No','TestNo'], ascending=[True, True]).reset_index(drop=True)
        # 패턴 list 및 선택 초기화
        self.ptn_list.clear()
        self.ptn_df_select = []
        # dataframe을 기준으로 table widget 생성
        self.ptn_list.setRowCount(len(self.pne_ptn_merged_df.index))
        self.ptn_list.setColumnCount(len(self.pne_ptn_merged_df.columns) - 3)
        self.ptn_list.setHorizontalHeaderLabels(["패턴폴더", "패턴이름", "비고"])
        self.ptn_list.horizontalHeader().setVisible(True)
        for row_index, row in enumerate(self.pne_ptn_merged_df.index):
            for col_index, column in enumerate(self.pne_ptn_merged_df.columns):
                value = self.pne_ptn_merged_df.loc[row][column]
                # QTableWidget의 row_index 열, col_index 행에 들어갈 아이템을 생성
                item = QtWidgets.QTableWidgetItem(str(value))
                # 생성된 아이템을 위젯의 row_index, col_index (행, 열)에 배치
                self.ptn_list.setItem(row_index, col_index, item)
        self.ptn_list.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.ptn_list.cellClicked.connect(self.ptn_get_selected_items)
        # 커서 및 연결 닫기
        conn.close()
        self.progressBar.setValue(100)

    def ptn_get_selected_items(self, row, column):
        self.ptn_df_select = []
        for index in self.ptn_list.selectionModel().selectedRows():
            self.ptn_df_select.append(self.pne_ptn_merged_df.iloc[index.row(), 3])
        if len(self.ptn_df_select) == 0:
            self.ptn_df_select = [""]

# UI 실행
if __name__ == "__main__":
    # HiDPI 스케일링을 명시적으로 비활성화합니다.
    # os.environ['QT_ENABLE_HIGHDPI_SCALING'] = '0'
    # os.environ['QT_SCALE_FACTOR'] = '1.0'
    app = QtWidgets.QApplication(sys.argv)
    # 개인 글꼴 폰트 적용
    app.setFont(QtGui.QFont("Malgun gothic"))
    # app.setStyleSheet("background-color: #FFFFFF;")
    myWindow = WindowClass()
    myWindow.show()
    # app.exec_()
    sys.exit(app.exec())