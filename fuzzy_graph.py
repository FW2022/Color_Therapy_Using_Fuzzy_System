import skfuzzy as fuzz
import numpy as np
import cv2
import csv
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.cluster import KMeans

# 사용자 그림 이미지
image = cv2.imread("img/img3.jpg")

# 채널을 BGR -> RGB로 변경
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# height, width 통합
image = image.reshape((image.shape[0] * image.shape[1], 3))

# k-mean 알고리즘으로 이미지를 학습시킨다.
k = 5
clt = KMeans(n_clusters = k)
clt.fit(image)

print("******실제 clustering된 컬러값(주조색, 보조색 RGB값)******")
# 실제 clustering된 컬러값 (주조색, 보조색 RGB값)
for center in clt.cluster_centers_:
    print(center)

# 주조색 RGB값
r1, g1, b1 = clt.cluster_centers_[0][0], clt.cluster_centers_[0][1], clt.cluster_centers_[0][2]

# 보조색 RGB값
r2, g2, b2 = clt.cluster_centers_[1][0], clt.cluster_centers_[1][1], clt.cluster_centers_[1][2]

# RGB 픽셀값 -> HSV 값으로 변환
def rgb_to_hsv(r, g, b):
    # RGB값은 0~255 값을 가지기 때문에 0~1의 값으로 바꿔줌
    # 결과 HSV값도 0~1의 값을 가지게 된다.
    r, g, b = r / 255.0, g / 255.0, b / 255.0

    # R, G, B 중 최대값과 최소값을 저장
    cmax = max(r, g, b)
    cmin = min(r, g, b)
    # 최대값과 최소값의 차 델타값을 저장
    diff = cmax - cmin

    # H(Hue: 색상) 계산
    if cmax == cmin:
        h_value = 0

    elif cmax == r:
        h_value = (60 * ((g - b) / diff) + 360) % 360

    elif cmax == g:
        h_value = (60 * ((b - r) / diff) + 120) % 360

    elif cmax == b:
        h_value = (60 * ((r - g) / diff) + 240) % 360

    # S(Saturation: 채도) 계산
    if cmax == 0:
        s_value = 0
    else:
        s_value = (diff / cmax) * 100

    # V(Value: 명도) 계산
    v_value = cmax * 100
    return [h_value, s_value, v_value]

arr1 = rgb_to_hsv(r1, g1, b1)
arr2 = rgb_to_hsv(r2, g2, b2)

# arr1(주조색), arr2(보조색)의 H, S, V값
h1, s1, v1 = int(arr1[0]), int(arr1[1]), int(arr1[2])
h2, s2, v2 = int(arr2[0]), int(arr2[1]), int(arr2[2])

# 변환된 HSV값
print("\n******주조색 & 보조색 HSV값******")
print(f"주조색 HSV값: {h1, s1, v1}")
print(f"보조색 HSV값: {h2, s2, v2}\n")

# HSV 멤버십 함수
# h: 색상, s: 채도, v: 명도, pc: 주조색, dc: 보조색
h = np.arange(0, 356, 1)
s = np.arange(0, 101, 1)
v = np.arange(0, 101, 1)
pc = np.arange(0, 101, 1)
dc = np.arange(0, 101, 1)

R1 = fuzz.trapmf(h, [0, 0, 5, 10])
O1 = fuzz.trapmf(h, [5, 10, 35, 50])
Y1 = fuzz.trapmf(h, [35, 50, 70, 85])
G1 = fuzz.trapmf(h, [70, 85, 160, 165])
B1 = fuzz.trapmf(h, [160, 165, 265, 280])
P1 = fuzz.trapmf(h, [265, 280, 315, 330])
R1_1 = fuzz.trapmf(h, [315, 330, 355, 355])

LS = fuzz.trapmf(s, [0, 0, 15, 40])
HS = fuzz.trapmf(s, [15, 40, 100, 100])

LV = fuzz.trapmf(v, [0, 0, 10, 20])
MV = fuzz.trapmf(v, [10, 20, 60, 75])
HV = fuzz.trapmf(v, [60, 75, 100, 100])

BLACK = fuzz.trapmf(pc, [0, 0, 4, 6])
GRAY = fuzz.trapmf(pc, [4, 6, 15, 20])
RED = fuzz.trapmf(pc, [20, 25, 35, 40])
ORANGE = fuzz.trapmf(pc, [35, 40, 50, 55])
YELLOW = fuzz.trapmf(pc, [50, 55, 65, 70])
GREEN = fuzz.trapmf(pc, [65, 70, 80, 82])
BLUE = fuzz.trapmf(pc, [80, 82, 90, 93])
PURPLE = fuzz.trapmf(pc, [90, 93, 100, 100])

##################### 퍼지화하는 멤버십 함수 #####################
def hMemfunc(h_val):
    red_val = fuzz.interp_membership(h, R1, h_val)
    org_val = fuzz.interp_membership(h, O1, h_val)
    ylw_val = fuzz.interp_membership(h, Y1, h_val)
    grn_val = fuzz.interp_membership(h, G1, h_val)
    blu_val = fuzz.interp_membership(h, B1, h_val)
    ppl_val = fuzz.interp_membership(h, P1, h_val)
    red_val2 = fuzz.interp_membership(h, R1_1, h_val)
    return red_val, org_val, ylw_val, grn_val, blu_val, ppl_val, red_val2, \
           dict(red=red_val, orange=org_val, yellow=ylw_val, green=grn_val, blue=blu_val, purple=ppl_val, red2=red_val2)

def sMemfunc(s_val):
    lowS_val = fuzz.interp_membership(s, LS, s_val)
    highS_val = fuzz.interp_membership(s, HS, s_val)
    return lowS_val, highS_val, dict(low_S=lowS_val, high_S=highS_val)

def vMemfunc(v_val):
    lowV_val = fuzz.interp_membership(v, LV, v_val)
    midV_val = fuzz.interp_membership(v, MV, v_val)
    highV_val = fuzz.interp_membership(v, HV, v_val)
    return lowV_val, midV_val, highV_val, dict(low_V=lowV_val, mid_V=midV_val, high_V=highV_val)

# 주조색 멤버십 함수 대응값
print("******주조색 HSV 멤버십 함수 소속값******")
hR1, hO1, hY1, hG1, hB1, hP1, hR1_1, h1_dict = hMemfunc(h1)
print(f"주조색 Hue 영역 소속값: {h1_dict}")

sL, sH, s1_dict = sMemfunc(s1)
print(f"주조색 Saturaiton 영역 소속값: {s1_dict}")

vL, vM, vH, v1_dict = vMemfunc(v1)
print(f"주조색 Value 영역 소속값: {v1_dict}\n")

# 그래프 총 6개
fig, (ax0, ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=6, figsize=(8, 15))

# h1(색상) 그래프
ax0.plot(h, R1, color='red', linewidth=1.5, label='RED')
ax0.plot(h, O1, color='darkorange', linewidth=1.5, label='ORANGE')
ax0.plot(h, Y1, color='yellow', linewidth=1.5, label='YELLOW')
ax0.plot(h, G1, color='limegreen', linewidth=1.5, label='GREEN')
ax0.plot(h, B1, color='blue', linewidth=1.5, label='BLUE')
ax0.plot(h, P1, color='darkviolet', linewidth=1.5, label='PURPLE')
ax0.plot(h, R1_1, color='red', linewidth=1.5, label='RED')
ax0.set_title('Hue')
ax0.legend()

# s1(채도) 그래프
ax1.plot(s, LS, 'b', linewidth=1.5, label='LS')
ax1.plot(s, HS, 'r', linewidth=1.5, label='HS')
ax1.set_title('Saturation')
ax1.legend()

# v1(명도) 그래프
ax2.plot(v, LV, 'b', linewidth=1.5, label='LV')
ax2.plot(v, MV, 'g', linewidth=1.5, label='MV')
ax2.plot(v, HV, 'r', linewidth=1.5, label='HV')
ax2.set_title('Value')
ax2.legend()

# pc(주조색) 그래프
ax3.plot(pc, BLACK, color='black', linewidth=1.5, label='BLACK')
ax3.plot(pc, GRAY, color='gray', linewidth=1.5, label='GRAY')
ax3.plot(pc, RED, color='red', linewidth=1.5, label='RED')
ax3.plot(pc, ORANGE, color='darkorange', linewidth=1.5, label='ORANGE')
ax3.plot(pc, YELLOW, color='yellow', linewidth=1.5, label='YELLOW')
ax3.plot(pc, GREEN, color='limegreen', linewidth=1.5, label='GREEN')
ax3.plot(pc, BLUE, color='blue', linewidth=1.5, label='BLUE')
ax3.plot(pc, PURPLE, color='darkviolet', linewidth=1.5, label='PURPLE')
ax3.set_title('Primary Color')
ax3.legend()

# 규칙1: IF s is sL AND v is vL THEN pc is BLACK
PC1_rule1 = max(sL, vL)  # 규칙 1: 계산용
gr1_rule1 = np.fmin(PC1_rule1, BLACK)  # 규칙 1: 그래프용

# 규칙2: IF s is sL AND v is vM THEN pc is GRAY
PC1_rule2 = min(sL, vM)  # 규칙 2: 계산용
gr1_rule2 = np.fmin(PC1_rule2, GRAY)  # 규칙 2: 그래프용

# 규칙3: IF h is hR1 AND s is sH AND v is vH THEN pc is RED
PC1_rule3 = min(hR1, sH, vH)  # 규칙 3: 계산용
gr1_rule3 = np.fmin(PC1_rule3, RED)  # 규칙 3: 그래프용

# 규칙3-1: IF h is hR1_1 AND s is sH AND v is vH THEN pc is RED
PC1_rule3_1 = min(hR1_1, sH, vH)  # 규칙 3-1: 계산용
gr1_rule3_1 = np.fmin(PC1_rule3_1, RED)  # 규칙 3-1: 그래프용

# 규칙4: IF h is hO1 AND s is sH AND v is vH THEN pc is ORANGE
PC1_rule4 = min(hO1, sH, vH)  # 규칙 4: 계산용
gr1_rule4 = np.fmin(PC1_rule4, ORANGE)  # 규칙 4 : 그래프용

# 규칙5: IF h is hY1 AND s is sH AND v is vH THEN pc is YELLOW
PC1_rule5 = min(hY1, sH, vH)  # 규칙 5: 계산용
gr1_rule5 = np.fmin(PC1_rule5, YELLOW)  # 규칙 5: 그래프용

# 규칙6: IF h is hG1 AND s is sH AND v is vH THEN pc is GREEN
PC1_rule6 = min(hG1, sH, vH)  # 규칙 6: 계산용
gr1_rule6 = np.fmin(PC1_rule6, GREEN)  # 규칙 6: 그래프용

# 규칙6-1: IF h is hG1 AND s is sH AND v is vM THEN pc is GREEN
PC1_rule6_1 = min(hG1, sH, vM)  # 규칙 6-1: 계산용
gr1_rule6_1 = np.fmin(PC1_rule6_1, GREEN)  # 규칙 6-1: 그래프용

# 규칙7: IF h is hB1 AND s is sH AND v is vH THEN pc is BLUE
PC1_rule7 = min(hB1, sH, vH)  # 규칙 7: 계산용
gr1_rule7 = np.fmin(PC1_rule7, BLUE)  # 규칙 7: 그래프용

# 규칙7-1: IF h is hB1 AND s is sH AND v is vM THEN pc is BLUE
PC1_rule7_1 = min(hB1, sH, vM)  # 규칙 7: 계산용
gr1_rule7_1 = np.fmin(PC1_rule7_1, BLUE)  # 규칙 7: 그래프용

# 규칙8: IF h is hP1 AND s is sH AND v is vH THEN pc is PURPLE
PC1_rule8 = min(hP1, sH, vH)  # 규칙 8: 계산용
gr1_rule8 = np.fmin(PC1_rule8, PURPLE)  # 규칙 8: 그래프용

# 규칙8-1: IF h is hP1 AND s is sH AND v is vM THEN pc is PURPLE
PC1_rule8_1 = min(hP1, sH, vM)  # 규칙 8: 계산용
gr1_rule8_1 = np.fmin(PC1_rule8_1, PURPLE)  # 규칙 8: 그래프용

pc0 = np.zeros_like(pc)

# 규칙 평가 합산 그래프
ax4.fill_between(pc, pc0, gr1_rule1, facecolor='b', alpha=0.7)
ax4.plot(pc, BLACK, color='black', linewidth=0.5, linestyle='--', )
ax4.fill_between(pc, pc0, gr1_rule2, facecolor='b', alpha=0.7)
ax4.plot(pc, GRAY, color='gray', linewidth=0.5, linestyle='--', )
ax4.fill_between(pc, pc0, gr1_rule3, facecolor='b', alpha=0.7)
ax4.plot(pc, RED, color='red', linewidth=0.5, linestyle='--', )
ax4.fill_between(pc, pc0, gr1_rule4, facecolor='b', alpha=0.7)
ax4.plot(pc, ORANGE, color='darkorange', linewidth=0.5, linestyle='--')
ax4.fill_between(pc, pc0, gr1_rule5, facecolor='b', alpha=0.7)
ax4.plot(pc, YELLOW, color='yellow', linewidth=0.5, linestyle='--')
ax4.fill_between(pc, pc0, gr1_rule6, facecolor='b', alpha=0.7)
ax4.plot(pc, GREEN, color='limegreen', linewidth=0.5, linestyle='--')
ax4.fill_between(pc, pc0, gr1_rule7, facecolor='b', alpha=0.7)
ax4.plot(pc, BLUE, color='blue', linewidth=0.5, linestyle='--')
ax4.fill_between(pc, pc0, gr1_rule8, facecolor='b', alpha=0.7)
ax4.plot(pc, PURPLE, color='darkviolet', linewidth=0.5, linestyle='--')
ax4.set_title('Rule evaluation')

# 규칙 후건의 통합 : 그래프용
aggregated1 = np.fmax(gr1_rule1, np.fmax(gr1_rule2, np.fmax(gr1_rule3, np.fmax(gr1_rule3_1, np.fmax(gr1_rule4, np.fmax(gr1_rule5, np.fmax(gr1_rule6, np.fmax(gr1_rule6_1,np.fmax(gr1_rule7, np.fmax(gr1_rule7_1, np.fmax(gr1_rule8, gr1_rule8_1)))))))))))

pc1 = fuzz.defuzz(pc, aggregated1, 'centroid')

print("******주조색 역퍼지값******")
print(f"주조색 역퍼지화 값: {pc1}\n")

# 역퍼지화 그래프
z_activation = fuzz.interp_membership(pc, aggregated1, pc1)
ax5.plot(pc, BLACK, color='black', linewidth=0.5, linestyle='--')
ax5.plot(pc, GRAY, color='gray', linewidth=0.5, linestyle='--')
ax5.plot(pc, RED, color='red', linewidth=0.5, linestyle='--')
ax5.plot(pc, ORANGE, color='darkorange', linewidth=0.5, linestyle='--')
ax5.plot(pc, YELLOW, color='yellow', linewidth=0.5, linestyle='--')
ax5.plot(pc, GREEN, color='limegreen', linewidth=0.5, linestyle='--')
ax5.plot(pc, BLUE, color='blue', linewidth=0.5, linestyle='--')
ax5.plot(pc, PURPLE, color='darkviolet', linewidth=0.5, linestyle='--')
ax5.fill_between(pc, pc0, aggregated1, facecolor='pink', alpha=0.8)
ax5.plot([pc1, pc1], [0, z_activation], 'k', linewidth=1.5, label='Input(PC1)={:.2f}'.format(pc1), alpha=0.9)
ax5.legend()
ax5.set_title('Defuzzification')

# 그래프 최종 시각화
ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.get_xaxis().tick_bottom()
ax0.get_yaxis().tick_left()
ax0.set_ylim([0, 1])
ax0.set_xlim([0, 355])
ax0.set_xticks([i for i in range(0, 356, 50)])

for ax in (ax1, ax2, ax3, ax4, ax5):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.set_ylim([0, 1])
    ax.set_xlim([0, 100])
    ax.set_xticks([i for i in range(0, 101, 10)])

# 보조색 멤버십 함수 대응값
print("******보조색 HSV 멤버십 함수 소속값******")
hR2, hO2, hY2, hG2, hB2, hP2, hR2_1, h2_dict = hMemfunc(h2)
print(f"보조색 Hue 영역 소속값: {h2_dict}")

sL2, sH2, s2_dict = sMemfunc(s2)
print(f"보조색 Saturation 영역 소속값: {s2_dict}")

vL2, vM2, vH2, v2_dict = vMemfunc(v2)
print(f"보조색 Value 영역 소속값: {v2_dict}\n")

DC1_rule1 = max(sL2, vL2)
DC1_rule2 = min(sL2, vM2)
DC1_rule3 = min(hR2, sH2, vH2)
DC1_rule3_1 = min(hR2_1, sH2, vH2)
DC1_rule4 = min(hO2, sH2, vH2)
DC1_rule5 = min(hY2, sH2, vH2)
DC1_rule6 = min(hG2, sH2, vH2)
DC1_rule6_1 = min(hG1, sH, vM)
DC1_rule7 = min(hB2, sH2, vH2)
DC1_rule7_1 = min(hB1, sH, vM)
DC1_rule8 = min(hP2, sH2, vH2)
DC1_rule8_1 = min(hP1, sH, vM)

gr2_rule1 = np.fmin(DC1_rule1, BLACK)
gr2_rule2 = np.fmin(DC1_rule2, GRAY)
gr2_rule3 = np.fmin(DC1_rule3, RED)
gr2_rule3_1 = np.fmin(DC1_rule3_1, RED)
gr2_rule4 = np.fmin(DC1_rule4, ORANGE)
gr2_rule5 = np.fmin(DC1_rule5, YELLOW)
gr2_rule6 = np.fmin(DC1_rule6, GREEN)
gr2_rule6_1 = np.fmin(PC1_rule6_1, GREEN)
gr2_rule7 = np.fmin(DC1_rule7, BLUE)
gr2_rule7_1 = np.fmin(PC1_rule7_1, BLUE)
gr2_rule8 = np.fmin(DC1_rule8, PURPLE)
gr2_rule8_1 = np.fmin(PC1_rule8_1, PURPLE)

aggregated2 = np.fmax(gr2_rule1, np.fmax(gr2_rule2, np.fmax(gr2_rule3, np.fmax(gr2_rule3_1, np.fmax(gr2_rule4, np.fmax(gr2_rule5, np.fmax(gr2_rule6, np.fmax(gr2_rule6_1, np.fmax(gr2_rule7,  np.fmax(gr2_rule7_1, np.fmax(gr2_rule8, gr2_rule8_1)))))))))))

dc1 = fuzz.defuzz(dc, aggregated2, 'centroid')
#dc1 = (DC1_rule1 + (10 + 20) * DC1_rule2 + 30 * DC1_rule3 + (40 + 50) * DC1_rule4 + 60 * DC1_rule5 + (70 + 80) * DC1_rule6 + 90 * DC1_rule7 + 100 * DC1_rule8) \
#     / ((DC1_rule1 * 2) + DC1_rule2 + DC1_rule3 + (DC1_rule4 * 2) + DC1_rule5 + (DC1_rule6 * 2) + DC1_rule7 + DC1_rule8)

print("******보조색 역퍼지화값******")
print(f"보조색 역퍼지화 값: {dc1}\n")

# 주조색-보조색 => 보완색
def rstColor(c1, c2):
    cc = []
    # 주조색: 검정
    if ((c1 >= 0) and (c1 < 6)):
        cc.append('검정')
        if ((c2 >= 6) and (c2 < 21)):  # 보조색: 회색
            cc.append('회색')
            cc.append('노랑')
        elif ((c2 >= 21) and (c2 < 38)):  # 보조색: 빨강
            cc.append('빨강')
            cc.append('파랑')
        elif ((c2 >= 38) and (c2 < 53)):  # 보조색: 주황
            cc.append('주황')
            cc.append('초록')
        elif ((c2 >= 53) and (c2 < 67)):  # 보조색: 노랑
            cc.append('노랑')
            cc.append('파랑')
        elif ((c2 >= 67) and (c2 < 82)):  # 보조색: 초록
            cc.append('초록')
            cc.append('파랑')
        elif ((c2 >= 82) and (c2 < 92)):  # 보조색: 파랑
            cc.append('파랑')
            cc.append('빨강')
        elif ((c2 >= 92) and (c2 <= 100)):  # 보조색: 보라
            cc.append('보라')
            cc.append('노랑')
        else:
            print('주조색과 보조색이 같음')

    # 주조색: 회색
    elif ((c1 >= 6) and (c1 < 21)):
        cc.append('회색')
        if ((c2 >= 0) and (c2 < 6)):  # 보조색: 검정
            cc.append('검정')
            cc.append('노랑')
        elif ((c2 >= 21) and (c2 < 38)):  # 보조색: 빨강
            cc.append('빨강')
            cc.append('노랑')
        elif ((c2 >= 38) and (c2 < 53)):  # 보조색: 주황
            cc.append('주황')
            cc.append('빨강')
        elif ((c2 >= 53) and (c2 < 67)):  # 보조색: 노랑
            cc.append('노랑')
            cc.append('주황')
        elif ((c2 >= 67) and (c2 < 82)):  # 보조색: 초록
            cc.append('초록')
            cc.append('빨강')
        elif ((c2 >= 82) and (c2 < 92)):  # 보조색: 파랑
            cc.append('파랑')
            cc.append('노랑')
        elif ((c2 >= 92) and (c2 <= 100)):  # 보조색: 보라
            cc.append('보라')
            cc.append('초록')
        else:
            print('주조색과 보조색이 같음')

    # 주조색: 빨강
    elif ((c1 >= 21) and (c1 < 38)):
        cc.append('빨강')
        if ((c2 >= 0) and (c2 < 6)):  # 보조색: 검정
            cc.append('검정')
            cc.append('파랑')
        elif ((c2 >= 6) and (c2 < 21)):  # 보조색: 회색
            cc.append('회색')
            cc.append('노랑')
        elif ((c2 >= 38) and (c2 < 53)):  # 보조색: 주황
            cc.append('주황')
            cc.append('초록')
        elif ((c2 >= 53) and (c2 < 67)):  # 보조색: 노랑
            cc.append('노랑')
            cc.append('파랑')
        elif ((c2 >= 67) and (c2 < 82)):  # 보조색: 초록
            cc.append('초록')
            cc.append('보라')
        elif ((c2 >= 82) and (c2 < 92)):  # 보조색: 파랑
            cc.append('파랑')
            cc.append('초록')
        elif ((c2 >= 92) and (c2 <= 100)):  # 보조색: 보라
            cc.append('보라')
            cc.append('검정')
        else:
            print('주조색과 보조색이 같음')

    # 주조색: 주황
    elif ((c1 >= 38) and (c1 < 53)):
        cc.append('주황')
        if ((c2 >= 0) and (c2 < 6)):  # 보조색: 검정
            cc.append('검정')
            cc.append('초록')
        elif ((c2 >= 6) and (c2 < 21)):  # 보조색: 회색
            cc.append('회색')
            cc.append('빨강')
        elif ((c2 >= 21) and (c2 < 38)):  # 보조색: 빨강
            cc.append('빨강')
            cc.append('초록')
        elif ((c2 >= 53) and (c2 < 67)):  # 보조색: 노랑
            cc.append('노랑')
            cc.append('검정')
        elif ((c2 >= 67) and (c2 < 82)):  # 보조색: 초록
            cc.append('초록')
            cc.append('파랑')
        elif ((c2 >= 82) and (c2 < 92)):  # 보조색: 파랑
            cc.append('파랑')
            cc.append('보라')
        elif ((c2 >= 92) and (c2 <= 100)):  # 보조색: 보라
            cc.append('보라')
            cc.append('파랑')
        else:
            print('주조색과 보조색이 같음')

    # 주조색: 노랑
    elif ((c1 >= 53) and (c1 < 67)):
        cc.append('노랑')
        if ((c2 >= 0) and (c2 < 6)):  # 보조색: 검정
            cc.append('검정')
            cc.append('파랑')
        elif ((c2 >= 6) and (c2 < 21)):  # 보조색: 회색
            cc.append('회색')
            cc.append('주황')
        elif ((c2 >= 21) and (c2 < 38)):  # 보조색: 빨강
            cc.append('빨강')
            cc.append('파랑')
        elif ((c2 >= 38) and (c2 < 53)):  # 보조색: 주황
            cc.append('주황')
            cc.append('검정')
        elif ((c2 >= 67) and (c2 < 82)):  # 보조색: 초록
            cc.append('초록')
            cc.append('검정')
        elif ((c2 >= 82) and (c2 < 92)):  # 보조색: 파랑
            cc.append('파랑')
            cc.append('보라')
        elif ((c2 >= 92) and (c2 <= 100)):  # 보조색: 보라
            cc.append('보라')
            cc.append('초록')
        else:
            print('주조색과 보조색이 같음')

    # 주조색: 초록
    elif ((c1 >= 67) and (c1 < 82)):
        cc.append('초록')
        if ((c2 >= 0) and (c2 < 6)):  # 보조색: 검정
            cc.append('검정')
            cc.append('파랑')
        elif ((c2 >= 6) and (c2 < 21)):  # 보조색: 회색
            cc.append('회색')
            cc.append('빨강')
        elif ((c2 >= 21) and (c2 < 38)):  # 보조색: 빨강
            cc.append('빨강')
            cc.append('보라')
        elif ((c2 >= 38) and (c2 < 53)):  # 보조색: 주황
            cc.append('주황')
            cc.append('파랑')
        elif ((c2 >= 53) and (c2 < 67)):  # 보조색: 노랑
            cc.append('노랑')
            cc.append('검정')
        elif ((c2 >= 82) and (c2 < 92)):  # 보조색: 파랑
            cc.append('파랑')
            cc.append('빨강')
        elif ((c2 >= 92) and (c2 <= 100)):  # 보조색: 보라
            cc.append('보라')
            cc.append('빨강')
        else:
            print('주조색과 보조색이 같음')

    # 주조색: 파랑
    elif ((c1 >= 82) and (c1 < 92)):
        cc.append('파랑')
        if ((c2 >= 0) and (c2 < 6)):  # 보조색: 검정
            cc.append('검정')
            cc.append('빨강')
        elif ((c2 >= 6) and (c2 < 21)):  # 보조색: 회색
            cc.append('회색')
            cc.append('노랑')
        elif ((c2 >= 21) and (c2 < 38)):  # 보조색: 빨강
            cc.append('빨강')
            cc.append('초록')
        elif ((c2 >= 38) and (c2 < 53)):  # 보조색: 주황
            cc.append('주황')
            cc.append('보라')
        elif ((c2 >= 53) and (c2 < 67)):  # 보조색: 노랑
            cc.append('노랑')
            cc.append('보라')
        elif ((c2 >= 67) and (c2 < 82)):  # 보조색: 초록
            cc.append('초록')
            cc.append('빨강')
        elif ((c2 >= 92) and (c2 <= 100)):  # 보조색: 보라
            cc.append('보라')
            cc.append('주황')
        else:
            print('주조색과 보조색이 같음')

    # 주조색: 보라
    elif ((c1 >= 92) and (c1 <= 100)):
        cc.append('보라')
        if ((c2 >= 0) and (c2 < 6)):  # 보조색: 검정
            cc.append('검정')
            cc.append('노랑')
        elif ((c2 >= 6) and (c2 < 21)):  # 보조색: 회색
            cc.append('회색')
            cc.append('초록')
        elif ((c2 >= 21) and (c2 < 38)):  # 보조색: 빨강
            cc.append('빨강')
            cc.append('검정')
        elif ((c2 >= 38) and (c2 < 53)):  # 보조색: 주황
            cc.append('주황')
            cc.append('파랑')
        elif ((c2 >= 53) and (c2 < 67)):  # 보조색: 노랑
            cc.append('노랑')
            cc.append('초록')
        elif ((c2 >= 67) and (c2 < 82)):  # 보조색: 초록
            cc.append('초록')
            cc.append('빨강')
        elif ((c2 >= 82) and (c2 < 92)):  # 보조색: 파랑
            cc.append('파랑')
            cc.append('주황')
        else:
            print('주조색과 보조색이 같음')

    return cc

cc = rstColor(pc1, dc1)
print("******주조색과 보조색에 따른 보완색******")
print(f"주조색: {cc[0]}, 보조색: {cc[1]}, 보완색: {cc[2]}")

# csv 쓰기
with open('result.csv', 'w', encoding='utf-8', newline='') as csvfile:
	writer = csv.writer(csvfile)
	writer.writerow(["주조색", "보조색", "보완색"])
	writer.writerow(cc)

plt.tight_layout()
#plt.savefig('img/fig1.jpg')
plt.show()
