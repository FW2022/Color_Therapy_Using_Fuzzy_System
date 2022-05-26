import skfuzzy as fuzz
import numpy as np
import cv2
import csv
from sklearn.cluster import KMeans

# 사용자 그림 이미지
image = cv2.imread("img/000.jpeg")

# 채널을 BGR -> RGB로 변경
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# height, width 통합
image = image.reshape((image.shape[0] * image.shape[1], 3))

# k-mean 알고리즘으로 이미지를 학습시킨다.
k = 4
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

# 보조색2 RGB값
r3, g3, b3 = clt.cluster_centers_[2][0], clt.cluster_centers_[2][1], clt.cluster_centers_[2][2]
r4, g4, b4 = clt.cluster_centers_[3][0], clt.cluster_centers_[3][1], clt.cluster_centers_[3][2]

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
arr3 = rgb_to_hsv(r3, g3, b3)
arr4 = rgb_to_hsv(r4, g4, b4)

# arr1(주조색), arr2(보조색)의 H, S, V값
h1, s1, v1 = int(arr1[0]), int(arr1[1]), int(arr1[2])
h2, s2, v2 = int(arr2[0]), int(arr2[1]), int(arr2[2])
h3, s3, v3 = int(arr3[0]), int(arr3[1]), int(arr3[2])
h4, s4, v4 = int(arr4[0]), int(arr4[1]), int(arr4[2])

# 변환된 HSV값
print("\n******주조색 & 보조색 HSV값******")
print(f"주조색 HSV값: {h1, s1, v1}")
print(f"보조색1 HSV값: {h2, s2, v2}")
print(f"보조색2 HSV값: {h3, s3, v3}")
print(f"보조색3 HSV값: {h4, s4, v4}\n")

# HSV 멤버십 함수
# h: 색상, s: 채도, v: 명도, pc: 주조색, dc: 보조색
h = np.arange(0, 361, 1)
s = np.arange(0, 101, 1)
v = np.arange(0, 101, 1)
pc = np.arange(0, 101, 1)
dc = np.arange(0, 101, 1)

R1 = fuzz.trapmf(h, [0, 0, 5, 10])
O1 = fuzz.trapmf(h, [5, 10, 35, 50])
Y1 = fuzz.trapmf(h, [35, 50, 70, 85])
G1 = fuzz.trapmf(h, [70, 85, 160, 165])
B1 = fuzz.trapmf(h, [160, 165, 250, 265])
P1 = fuzz.trapmf(h, [250, 265, 315, 330])
R1_1 = fuzz.trapmf(h, [315, 330, 360, 360])

LS = fuzz.trapmf(s, [0, 0, 15, 45])
HS = fuzz.trapmf(s, [15, 40, 100, 100])

LV = fuzz.trapmf(v, [0, 0, 10, 35])
MV = fuzz.trapmf(v, [10, 30, 60, 75])
HV = fuzz.trapmf(v, [55, 75, 100, 100])

BLACK = fuzz.trapmf(pc, [0, 0, 4, 7])
GRAY = fuzz.trapmf(pc, [4, 7, 15, 20])
RED = fuzz.trapmf(pc, [20, 22, 33, 35])
ORANGE = fuzz.trapmf(pc, [33, 40, 50, 55])
YELLOW = fuzz.trapmf(pc, [50, 52, 60, 63])
GREEN = fuzz.trapmf(pc, [60, 63, 72, 75])
BLUE = fuzz.trapmf(pc, [72, 75, 86, 90])
PURPLE = fuzz.trapmf(pc, [86, 90, 100, 100])

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

def colorDefuzzVal(h, s, v):
    hR1, hO1, hY1, hG1, hB1, hP1, hR1_1, h1_dict = hMemfunc(h)
    sL, sH, s1_dict = sMemfunc(s)
    vL, vM, vH, v1_dict = vMemfunc(v)

    # 규칙1: IF s is sL AND v is vL THEN pc is BLACK
    PC1_rule1 = min(sL, vL)  # 규칙 1: 계산용
    gr1_rule1 = np.fmin(PC1_rule1, BLACK)  # 규칙 1: 그래프용

    # 규칙2: IF s is sL AND v is vM THEN pc is GRAY
    PC1_rule2 = min(sL, vM)  # 규칙 2: 계산용
    gr1_rule2 = np.fmin(PC1_rule2, GRAY)  # 규칙 2: 그래프용
    #
    # # 규칙2-1: IF s is sL AND v is vH THEN pc is GRAY
    # PC1_rule2_1 = min(sL, vH)  # 규칙 2_1: 계산용
    # gr1_rule2_1 = np.fmin(PC1_rule2_1, GRAY)  # 규칙 2_1: 그래프용

    # 규칙3: IF h is hR1 AND s is sH AND v is vH THEN pc is RED
    PC1_rule3 = min(hR1, sH, vH)  # 규칙 3: 계산용
    gr1_rule3 = np.fmin(PC1_rule3, RED)  # 규칙 3: 그래프용

    # 규칙3-1: IF h is hR1_1 AND s is sH AND v is vH THEN pc is RED
    PC1_rule3_1 = min(hR1_1, sH, vH)  # 규칙 3: 계산용
    gr1_rule3_1 = np.fmin(PC1_rule3_1, RED)  # 규칙 3: 그래프용

    # 규칙3-2: IF h is hR1 AND s is sH AND v is vM THEN pc is RED
    PC1_rule3_2 = min(hR1, sH, vM)  # 규칙 3-2: 계산용
    gr1_rule3_2 = np.fmin(PC1_rule3_2, RED)  # 규칙 3-2: 그래프용

    # 규칙3-3: IF h is hR1_1 AND s is sH AND v is vM THEN pc is RED
    PC1_rule3_3 = min(hR1_1, sH, vM)  # 규칙 3-3: 계산용
    gr1_rule3_3 = np.fmin(PC1_rule3_3, RED)  # 규칙 3-3: 그래프용

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

    # 규칙6-2: IF h is hG1 AND s is sL AND v is vH THEN pc is GREEN
    PC1_rule6_2 = min(hG1, sL, vH)  # 규칙 6-2: 계산용
    gr1_rule6_2 = np.fmin(PC1_rule6_2, GREEN)  # 규칙 6-2: 그래프용

    # 규칙7: IF h is hB1 AND s is sH AND v is vH THEN pc is BLUE
    PC1_rule7 = min(hB1, sH, vH)  # 규칙 7: 계산용
    gr1_rule7 = np.fmin(PC1_rule7, BLUE)  # 규칙 7: 그래프용

    # 규칙7-1: IF h is hB1 AND s is sH AND v is vM THEN pc is BLUE
    PC1_rule7_1 = min(hB1, sH, vM)  # 규칙 7-1: 계산용
    gr1_rule7_1 = np.fmin(PC1_rule7_1, BLUE)  # 규칙 7-1: 그래프용

    # 규칙7-2: IF h is hB1 AND s is sH AND v is vM THEN pc is BLUE
    PC1_rule7_2 = min(hB1, sH, vM)  # 규칙 7-2: 계산용
    gr1_rule7_2 = np.fmin(PC1_rule7_2, BLUE)  # 규칙 7-2: 그래프용

    # 규칙8: IF h is hP1 AND s is sH AND v is vH THEN pc is PURPLE
    PC1_rule8 = min(hP1, sH, vH)  # 규칙 8: 계산용
    gr1_rule8 = np.fmin(PC1_rule8, PURPLE)  # 규칙 8: 그래프용

    # 규칙8-1: IF h is hP1 AND s is sH AND v is vM THEN pc is PURPLE
    PC1_rule8_1 = min(hP1, sH, vM)  # 규칙 8: 계산용
    gr1_rule8_1 = np.fmin(PC1_rule8_1, PURPLE)  # 규칙 8: 그래프용

    # 규칙 후건의 통합 : 그래프용
    aggregated1 = np.fmax(gr1_rule1, np.fmax(gr1_rule2, np.fmax(gr1_rule3, np.fmax(gr1_rule3_1,
        np.fmax(gr1_rule3_2, np.fmax(gr1_rule3_3, np.fmax(gr1_rule4, np.fmax(gr1_rule5, np.fmax(gr1_rule6,
         np.fmax(gr1_rule6_1, np.fmax(gr1_rule6_2, np.fmax(gr1_rule7, np.fmax(gr1_rule7_1,
            np.fmax(gr1_rule7_2, np.fmax(gr1_rule8, gr1_rule8_1)))))))))))))))

    if (aggregated1.sum() != 0):
        pc1 = fuzz.defuzz(pc, aggregated1, 'centroid')
    else:
        pc1 = fuzz.defuzz(pc, aggregated1, 'mom')

    return pc1

c1 = colorDefuzzVal(h1, s1, v1)
c2 = colorDefuzzVal(h2, s2, v2)
c3 = colorDefuzzVal(h3, s3, v3)
c4 = colorDefuzzVal(h4, s4, v4)

# 주조색-보조색 => 보완색
def rstColor(c1, c2):
    cc = []
    # 주조색: 검정
    if ((c1 >= 0) and (c1 < 8)):
        cc.append('검정')
        if ((c2 >= 8) and (c2 < 20)):  # 보조색: 회색
            cc.append('회색')
            cc.append('노랑')
        elif ((c2 >= 20) and (c2 < 36)):  # 보조색: 빨강
            cc.append('빨강')
            cc.append('파랑')
        elif ((c2 >= 36) and (c2 < 51)):  # 보조색: 주황
            cc.append('주황')
            cc.append('초록')
        elif ((c2 >= 51) and (c2 < 64)):  # 보조색: 노랑
            cc.append('노랑')
            cc.append('파랑')
        elif ((c2 >= 64) and (c2 < 76)):  # 보조색: 초록
            cc.append('초록')
            cc.append('파랑')
        elif ((c2 >= 77) and (c2 < 91)):  # 보조색: 파랑
            cc.append('파랑')
            cc.append('빨강')
        elif ((c2 >= 91) and (c2 <= 100)):  # 보조색: 보라
            cc.append('보라')
            cc.append('노랑')
        elif ((c2 >= 0) and (c2 < 8)):
            print("동일")

    # 주조색: 회색
    elif ((c1 >= 8) and (c1 < 20)):
        cc.append('회색')
        if ((c2 >= 0) and (c2 < 8)):  # 보조색: 검정
            cc.append('검정')
            cc.append('노랑')
        elif ((c2 >= 20) and (c2 < 36)):  # 보조색: 빨강
            cc.append('빨강')
            cc.append('노랑')
        elif ((c2 >= 36) and (c2 < 51)):  # 보조색: 주황
            cc.append('주황')
            cc.append('빨강')
        elif ((c2 >= 51) and (c2 < 64)):  # 보조색: 노랑
            cc.append('노랑')
            cc.append('주황')
        elif ((c2 >= 64) and (c2 < 76)):  # 보조색: 초록
            cc.append('초록')
            cc.append('빨강')
        elif ((c2 >= 77) and (c2 < 91)):  # 보조색: 파랑
            cc.append('파랑')
            cc.append('노랑')
        elif ((c2 >= 91) and (c2 <= 100)):  # 보조색: 보라
            cc.append('보라')
            cc.append('초록')
        elif ((c2 >= 8) and (c2 < 20)):
            print("동일")

    # 주조색: 빨강
    elif ((c1 >= 20) and (c1 < 36)):
        cc.append('빨강')
        if ((c2 >= 0) and (c2 < 8)):  # 보조색: 검정
            cc.append('검정')
            cc.append('파랑')
        elif ((c2 >= 8) and (c2 < 20)):  # 보조색: 회색
            cc.append('회색')
            cc.append('노랑')
        elif ((c2 >= 36) and (c2 < 51)):  # 보조색: 주황
            cc.append('주황')
            cc.append('초록')
        elif ((c2 >= 51) and (c2 < 64)):  # 보조색: 노랑
            cc.append('노랑')
            cc.append('파랑')
        elif ((c2 >= 64) and (c2 < 76)):  # 보조색: 초록
            cc.append('초록')
            cc.append('보라')
        elif ((c2 >= 77) and (c2 < 91)):  # 보조색: 파랑
            cc.append('파랑')
            cc.append('초록')
        elif ((c2 >= 91) and (c2 <= 100)):  # 보조색: 보라
            cc.append('보라')
            cc.append('검정')
        elif ((c2 >= 20) and (c2 < 36)):
            print("동일")

    # 주조색: 주황
    elif ((c1 >= 36) and (c1 < 51)):
        cc.append('주황')
        if ((c2 >= 0) and (c2 < 8)):  # 보조색: 검정
            cc.append('검정')
            cc.append('초록')
        elif ((c2 >= 8) and (c2 < 20)):  # 보조색: 회색
            cc.append('회색')
            cc.append('빨강')
        elif ((c2 >= 20) and (c2 < 36)):  # 보조색: 빨강
            cc.append('빨강')
            cc.append('초록')
        elif ((c2 >= 51) and (c2 < 64)):  # 보조색: 노랑
            cc.append('노랑')
            cc.append('검정')
        elif ((c2 >= 64) and (c2 < 76)):  # 보조색: 초록
            cc.append('초록')
            cc.append('파랑')
        elif ((c2 >= 77) and (c2 < 91)):  # 보조색: 파랑
            cc.append('파랑')
            cc.append('보라')
        elif ((c2 >= 91) and (c2 <= 100)):  # 보조색: 보라
            cc.append('보라')
            cc.append('파랑')
        elif ((c2 >= 36) and (c2 < 51)):
            print("동일")

    # 주조색: 노랑
    elif ((c1 >= 51) and (c1 < 64)):
        cc.append('노랑')
        if ((c2 >= 0) and (c2 < 8)):  # 보조색: 검정
            cc.append('검정')
            cc.append('파랑')
        elif ((c2 >= 8) and (c2 < 20)):  # 보조색: 회색
            cc.append('회색')
            cc.append('주황')
        elif ((c2 >= 20) and (c2 < 36)):  # 보조색: 빨강
            cc.append('빨강')
            cc.append('파랑')
        elif ((c2 >= 36) and (c2 < 51)):  # 보조색: 주황
            cc.append('주황')
            cc.append('검정')
        elif ((c2 >= 64) and (c2 < 76)):  # 보조색: 초록
            cc.append('초록')
            cc.append('검정')
        elif ((c2 >= 77) and (c2 < 91)):  # 보조색: 파랑
            cc.append('파랑')
            cc.append('보라')
        elif ((c2 >= 91) and (c2 <= 100)):  # 보조색: 보라
            cc.append('보라')
            cc.append('초록')
        elif ((c2 >= 51) and (c2 < 64)):
            print("동일")

    # 주조색: 초록
    elif ((c1 >= 64) and (c1 < 76)):
        cc.append('초록')
        if ((c2 >= 0) and (c2 < 8)):  # 보조색: 검정
            cc.append('검정')
            cc.append('파랑')
        elif ((c2 >= 8) and (c2 < 20)):  # 보조색: 회색
            cc.append('회색')
            cc.append('빨강')
        elif ((c2 >= 20) and (c2 < 36)):  # 보조색: 빨강
            cc.append('빨강')
            cc.append('보라')
        elif ((c2 >= 36) and (c2 < 51)):  # 보조색: 주황
            cc.append('주황')
            cc.append('파랑')
        elif ((c2 >= 51) and (c2 < 64)):  # 보조색: 노랑
            cc.append('노랑')
            cc.append('검정')
        elif ((c2 >= 77) and (c2 < 91)):  # 보조색: 파랑
            cc.append('파랑')
            cc.append('빨강')
        elif ((c2 >= 91) and (c2 <= 100)):  # 보조색: 보라
            cc.append('보라')
            cc.append('빨강')
        elif ((c2 >= 64) and (c2 < 76)):
            print("동일")

    # 주조색: 파랑
    elif ((c1 >= 77) and (c1 < 91)):
        cc.append('파랑')
        if ((c2 >= 0) and (c2 < 8)):  # 보조색: 검정
            cc.append('검정')
            cc.append('빨강')
        elif ((c2 >= 8) and (c2 < 20)):  # 보조색: 회색
            cc.append('회색')
            cc.append('노랑')
        elif ((c2 >= 20) and (c2 < 36)):  # 보조색: 빨강
            cc.append('빨강')
            cc.append('초록')
        elif ((c2 >= 36) and (c2 < 51)):  # 보조색: 주황
            cc.append('주황')
            cc.append('보라')
        elif ((c2 >= 51) and (c2 < 64)):  # 보조색: 노랑
            cc.append('노랑')
            cc.append('보라')
        elif ((c2 >= 64) and (c2 < 76)):  # 보조색: 초록
            cc.append('초록')
            cc.append('빨강')
        elif ((c2 >= 91) and (c2 <= 100)):  # 보조색: 보라
            cc.append('보라')
            cc.append('주황')
        elif ((c2 >= 77) and (c2 < 91)):
            print("동일")

    # 주조색: 보라
    elif ((c1 >= 91) and (c1 <= 100)):
        cc.append('보라')
        if ((c2 >= 0) and (c2 < 8)):  # 보조색: 검정
            cc.append('검정')
            cc.append('노랑')
        elif ((c2 >= 8) and (c2 < 20)):  # 보조색: 회색
            cc.append('회색')
            cc.append('초록')
        elif ((c2 >= 20) and (c2 < 36)):  # 보조색: 빨강
            cc.append('빨강')
            cc.append('검정')
        elif ((c2 >= 36) and (c2 < 51)):  # 보조색: 주황
            cc.append('주황')
            cc.append('파랑')
        elif ((c2 >= 51) and (c2 < 64)):  # 보조색: 노랑
            cc.append('노랑')
            cc.append('초록')
        elif ((c2 >= 64) and (c2 < 76)):  # 보조색: 초록
            cc.append('초록')
            cc.append('빨강')
        elif ((c2 >= 77) and (c2 < 91)):  # 보조색: 파랑
            cc.append('파랑')
            cc.append('주황')
        elif ((c2 >= 91) and (c2 < 100)):
            print("동일")
    return cc

cc1 = rstColor(c1, c2)
cc2 = rstColor(c1, c3)
cc3 = rstColor(c1, c4)

print(cc1)
print(cc2)
print(cc3)

if (len(cc1) == 3):
    rst = cc1
elif (len(cc2) == 3):
    rst = cc2
elif (len(cc3) == 3):
    rst = cc3

# csv 쓰기
with open('result.csv', 'w', encoding='utf-8', newline='') as csvfile:
	writer = csv.writer(csvfile)
	writer.writerow(["주조색", "보조색", "보완색"])
	writer.writerow(rst)

