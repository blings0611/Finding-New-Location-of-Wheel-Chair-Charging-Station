### STEP1) Import Libraries

import pandas as pd
from pandas import DataFrame
import numpy as np
import requests
from bs4 import BeautifulSoup
import os
from datetime import datetime



### STEP2) shp(서울시 동별) csv 데이터
데이터 파일명 : <DATA_LSMD_ADM_SECT_UMD_서울_동별_prc>
shp(서울시 동별) 데이터는 pca/clustering 분석에 필요한 데이터 셋의 기준점이 됨 (동 명, 동 코드 등)

dong_shp = pd.read_csv('DATA_LSMD_ADM_SECT_UMD_서울_동별_prc.csv', encoding = 'cp949')
dong_shp_df = dong_shp.iloc[:, :2]
dong_shp_df.columns = ['EMD_CD', 'EMD_NM']
dong_shp_df



### STEP3) 장애인콜택시 이용고객 목적지(동단위) Best100 데이터
#### STEP3 - 1) 장애인콜택시 이용고객 목적지(동단위) Best100 - 크롤링
해당 데이터는 OpenAPI 형태로 데이터포맷은 JSON+XML 형식
2019/01/01 ~ 2019/12/31 데이터 수집 위해 크롤링 진행

#수집 기간 정의
import datetime

days_range = []

start = datetime.datetime.strptime("20190101", "%Y%m%d") #시작점 : 2019.01.01
end = datetime.datetime.strptime("20200101", "%Y%m%d") #끝점 : 2019.12.31
date_generated = [start + datetime.timedelta(days = x) for x in range(0, (end-start).days)]

for date in date_generated:
    days_range.append(date.strftime("%Y%m%d"))

#크롤러 작성
from tqdm.notebook import tqdm

temp = []

#데이터 불러오기
def get_data(url):
    response = requests.get(url).text
    df = pd.read_html(response)
    return df

#(날짜별) 포맷에 맞춰 url 크롤링
for date in tqdm(days_range):
    base_url = f'http://m.calltaxi.sisul.or.kr/api/open/newEXCEL0002.asp?key=29c3729265fa3b49c7d3e75147dc19a3&sDate={date}'
    df1 = get_data(base_url)
    
    for row in df1:
        row
    
    temp.append(row)

# 날짜별 데이터 하나의 데이터셋으로 저장
taxi_raw = pd.concat(temp, ignore_index = True)
taxi_raw

taxi_raw.to_csv('DATA_장애인콜택시 이용고객 목적지(동단위) Best100_raw.csv', encoding = 'cp949')

# 데이터 파일명 : <DATA_장애인콜택시 이용고객 목적지(동단위) Best100_raw>
# =>20190101 ~ 20191231 raw데이터 저장 완료


#### STEP3 - 2) 장애인콜택시 이용고객 목적지(동단위) Best100 전처리 (1st)
STEP3 - 1)에서 수집한 Raw Data 전처리 진행
Raw Data는 전국 데이터 포함 -> '서울특별시'만 추출하여 동별 택시 건수 탐색

# taxi 데이터 칼럼명 변경
taxi = taxi_raw.iloc[:]
taxi.columns = ['DATE', 'SI_NM', 'GU_NM', 'EMD_NM', 'TAXI_CNT']

# raw데이터에서 '서울특별시' 데이터만 출력해서 새로운 csv파일 저장
taxi_seoul = taxi[taxi['SI_NM'] == '서울특별시']
taxi_seoul.set_index("DATE", inplace = True)
taxi_seoul.to_csv('DATA_장애인콜택시 이용고객 목적지(동단위) Best100_Seoul_prc.csv', encoding = 'cp949')

# 동별 택시 건수 더해서 새로운 csv 파일 저장
taxi_seoul_df = pd.read_csv('DATA_장애인콜택시 이용고객 목적지(동단위) Best100_Seoul_prc.csv', encoding = 'cp949')
taxi_seoul_df2 = taxi_seoul_df.iloc[:, 3:]
taxi_dong = taxi_seoul_df2.groupby('EMD_NM').sum()
taxi_dong
taxi_dong.to_csv('DATA_장애인콜택시 이용고객 목적지(동단위) Best100_Seoul_dong_prc.csv', encoding = 'cp949')


#### STEP3 - 3) 동별 택시 건수 데이터는 행정동+법정동 구성 -> 법정동 기준으로 변경
데이터 파일명 : <DATA_장애인콜택시 이용고객 목적지(동단위) Best100_Seoul_dong_(법정동수정)_prc>
* [법정동 변경 기준]
sol1) 행정동을 법정동으로 변경
sol2) 하나의 행정동이 여러 개의 법정동으로 나눠질 경우 '주민센터' 기준 법정동으로 변경


#### STEP3 - 4) 장애인콜택시 이용고객 목적지(동단위) Best100 전처리 (2nd)

# 행정동 -> 법정동 변경 완료된 데이터 다시 전처리 진행
taxi_prc = pd.read_csv('DATA_장애인콜택시 이용고객 목적지(동단위) Best100_Seoul_dong_(법정동수정)_prc.csv', encoding = 'cp949')
taxi_prc_df = taxi_prc.iloc[:, :2]

taxi_prc_df2 = taxi_prc_df.groupby('EMD_NM').sum()
taxi_prc_df2

# shp(서울시 동별) 데이터 + 장애인콜택시 목적지 Best100 데이터 JOIN
법정동 기준으로 데이터 구성된 shp(서울시 동별) 데이터와 전처리 완료된 장애인콜택시 목적지 Best100 데이터 조인하여 최종 전처리 완료 데이터 csv파일 저장 (EMD_CD, EMD_NM, TAXI_CNT로 구성)
taxi_df = pd.merge(left = dong_shp_df, right = taxi_prc_df2, how = 'left', on = 'EMD_NM')
taxi_df = taxi_df.fillna(0)
taxi_df.set_index('EMD_CD', inplace = True)
taxi_df
taxi_df.to_csv('DATA_장애인콜택시 이용고객 목적지(동단위) Best100_prc.csv', encoding = 'cp949')

# 데이터 파일명 : <DATA_장애인콜택시 이용고객 목적지(동단위) Best100_prc> => '장애인콜택시 목적지 Best100' 최종 전처리 완료 데이터



### STEP4) 장애인 구인 현황
#### STEP4 - 1) 장애인 구인 현황 전처리 (1st)
데이터 파일명 : <DATA_장애인 구인 현황_EMD_NM_prc>
-> 위의 데이터는 <DATA_장애인 구인 현황_raw.csv> 데이터에 동 이름과 위/경도 추가한 데이터

# 장애인 동별 구인현황 데이터 불러오기
job = pd.read_csv("DATA_장애인 구인 현황_EMD_NM_prc.csv", encoding = 'cp949')

# raw 데이터에서 동별 구인현황 추출
job_dong = job.groupby('EMD_NM')
job_dong_sum = job_dong['구인인원'].sum()
job_dong_df = pd.DataFrame(job_dong_sum)

# shp(서울시 동별) 데이터와 조인 후 저장
job_join = pd.merge(left = dong_shp_df, right = job_dong_df, how = 'right', on = 'EMD_NM')
job_join.to_csv("DATA_장애인 구인 현황_EMD_NM_(shp)1stJOIN_prc.csv", encoding = 'cp949')


#### STEP4 - 2) 장애인 구인 현황 데이터는 행정동+법정동 구성 -> 법정동 기준으로 변경
데이터 파일명 : <DATA_장애인 구인 현황_EMD_NM_(shp)1stJOIN_(법정동수정)_prc>
* [법정동 변경 기준]
sol1) 행정동을 법정동으로 변경
sol2) 하나의 행정동이 여러 개의 법정동으로 나눠질 경우 '주민센터' 기준 법정동으로 변경


#### STEP4 - 3) 장애인 구인 현황 전처리 (2nd)

# 행정동 -> 법정동 변경 완료된 데이터 다시 전처리 진행
job_prc = pd.read_csv('DATA_장애인 구인 현황_EMD_NM_(shp)1stJOIN_(법정동수정)_prc.csv', encoding = 'cp949')
job_prc_df = job_prc.iloc[:, 1:3]

job_prc_df2 = job_prc_df.groupby('EMD_NM').sum()
job_prc_df2

# shp(서울시 동별) 데이터 + 장애인 구인 현황 데이터 JOIN

# #shp(동별) 데이터 기준 장애인 구인 현황 데이터 조인 및 저장
job_df = pd.merge(left = dong_shp_df, right = job_prc_df2, how = 'left', on = 'EMD_NM')
job_df = job_df.fillna(0)
job_df.set_index('EMD_CD', inplace = True)
job_df.to_csv("DATA_장애인 구인 현황_prc.csv", encoding = 'cp949')

# 데이터 파일명 : <DATA_장애인 구인 현황_prc> => '장애인 구인 현황' 최종 전처리 완료 데이터



### STEP5) 서울시 장애인 현황(장애유형별, 동별) 통계
데이터 파일명 : <DATA_서울시 장애인 현황(장애유형별, 동별) 통계_prc>
위의 데이터는 <DATA_서울시 장애인 현황(장애유형별,동별) 통계_raw.txt> 데이터를 '지체/뇌병변/심장장애/호흡기장애'만 추출하여 동 이름 추가하고 장애인 수 카운트한 데이터

disabled_df = pd.read_csv('DATA_서울시 장애인 현황(장애유형별, 동별) 통계_prc.csv', encoding = 'cp949')
disabled_df



### STEP6) 서울시 장애인시설 현황
데이터 파일명 : <DATA_서울시 장애인시설 현황_prc>
위의 데이터는 <DATA_서울시 장애인시설 현황_raw.csv> 데이터를 QGIS에서 시각화 후 법정동별 개수 카운트한 데이터

welfare_df = pd.read_csv('DATA_서울시 장애인시설 현황_prc.csv', encoding = 'cp949')
welfare_df



### STEP7) 서울시 신한카드 장애인 복지카드 이용현황
데이터 파일명 : <DATA_서울시 신한카드 장애인 복지카드 이용현황_prc>
위의 데이터는 '서울시 빅데이터 캠퍼스'가 보유한 Raw Data의 법정동별 이용횟수를 QGIS에서 시각화하여 반출. 이후 시각화한 자료를 카운트한 데이터

card_df = pd.read_csv('DATA_서울시 신한카드 장애인 복지카드 이용현황_prc.csv', encoding = 'cp949')
card_df



### STEP8) 대규모점포 인허가 정보
데이터 파일명 : <DATA_대규모점포 인허가 정보_prc>
위의 데이터는 <DATA_대규모점포 인허가 정보_raw.csv> 데이터를 QGIS에서 시각화 후 법정동별로 카운트한 데이터

shop_df =  pd.read_csv('DATA_대규모점포 인허가 정보_prc.csv', encoding = 'cp949')
shop_df



### STEP9) 활용데이터 전체 JOIN
pca/clustering 분석을 위한 하나의 데이터 셋으로 csv파일 저장

join1 = pd.merge(left = dong_shp_df, right = disabled_df.iloc[:, 1:], how = 'left', on = 'EMD_CD')
join2 = pd.merge(left = join1, right = shop_df.iloc[:, 1:], how = 'left', on = 'EMD_CD')
join3 = pd.merge(left = join2, right = taxi_df.iloc[:, 1:], how = 'left', on = "EMD_CD")
join4 = pd.merge(left = join3, right = job_df.iloc[:, 1:], how = 'left', on = 'EMD_CD')
join5 = pd.merge(left = join4, right = welfare_df.iloc[:, 1:], how = 'left', on = 'EMD_CD')
join6 = pd.merge(left = join5, right = card_df.iloc[:, 1:], how = 'left', on = 'EMD_CD')

# 결측치 처리 및 저장
total = join6.fillna(0)
total.set_index('EMD_CD', inplace = True)
total
total.to_csv('DATA_TOTAL DATASET.csv', encoding = 'cp949')