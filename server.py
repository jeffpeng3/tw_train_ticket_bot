# -*- coding: utf-8 -*-
import requests
from bs4 import BeautifulSoup
from datetime import datetime,timezone,timedelta
import threading
import captcha_handler # 導入新的驗證碼處理模組
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class BookingConfig:
    person_id: str
    from_station: str
    to_station: str
    getin_date: str
    train_no: List[str] = field(default_factory=lambda: ["", "", ""])
    ticket_num: int
    can_change_seat: bool = True
    normal_ticket_price: Optional[float] = None
    round_trip: bool = False
    return_date: Optional[str] = None
    multi_ride_filter: bool = False
    multi_ride_ticket_type: Optional[str] = None
    student_ticket_type: Optional[str] = None


class RailwayBot(threading.Thread):
  # 建構式
  def __init__(self, config: BookingConfig):
      threading.Thread.__init__(self)
      self.config = config
      self.session = requests.Session()
      self.session.headers.update({
          "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_4) AppleWebKit/" "537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
      })

      # 確保 train_no 列表至少有三個元素，不足則補空字串
      train_no_list = list(self.config.train_no) # 創建副本以修改
      while len(train_no_list) < 3:
          train_no_list.append("")

      train_no_0 = train_no_list[0]
      train_no_1 = train_no_list[1]
      train_no_2 = train_no_list[2]

      start_title = f"{self.now()}\n{self.config.getin_date} {self.config.from_station}👉{self.config.to_station} {self.config.ticket_num}張 {train_no_0} {train_no_1} {train_no_2}".strip()
      print(start_title)
      # self.main()

      self.kill_status = False


  def run(self):
    print("開始搶票！")
    self.main()

  def kill(self):
    self.kill_status = True
    print("已停止搶票！")




  def now(self):
    dt1 = datetime.utcnow().replace(tzinfo=timezone.utc)
    dt2 = dt1.astimezone(timezone(timedelta(hours=8))) # 轉換時區 -> 東八區
    return (dt2.strftime("%Y-%m-%d %H:%M:%S")) # 將時間轉換為 string

  def buy_tickets(self): # 移除參數
    response = self.session.get("https://www.railway.gov.tw/tra-tip-web/tip/tip001/tip121/query")
    # Session 會自動處理 cookies，所以不需要手動檢查 T4TIPSESSIONID 或傳遞 cookies

    # 準備獲取驗證碼
    # 根據 captcha_handler.py 的原始邏輯，需要先訪問 nonPicture URL
    # pageRandom 的值似乎是固定的，這裡我們沿用
    try:
        self.session.get("https://www.railway.gov.tw/tra-tip-web/tip/player/nonPicture?pageRandom=123")
    except requests.exceptions.RequestException as e:
        print(f"Error accessing nonPicture URL: {e}")
        return "訪問驗證碼前置頁面失敗"

    # 實際的語音驗證碼 URL，pageRandom 值也沿用原始 captcha_handler.py 中的固定值
    actual_audio_captcha_url = "https://www.railway.gov.tw/tra-tip-web/tip/player/audio?pageRandom=1079330974"

    # 調用新的 resolve_audio_captcha 函式
    verifyCode = captcha_handler.resolve_audio_captcha(self.session, actual_audio_captcha_url)

    if not verifyCode: # resolve_audio_captcha 在失敗時返回空字串
        print("Error: Failed to resolve captcha audio.")
        return "辨識驗證碼失敗"
    
    print(f"辨識出的驗證碼: {verifyCode}")

    html_doc = response.text # text 屬性就是 html 檔案
    soup = BeautifulSoup(html_doc, "lxml") # 指定 lxml 作為解析器

    # 資料
    soup_form = soup.find('form')
    if not soup_form:
        print("Error: Could not find form in booking page.")
        return "找不到訂票表單"

    _csrf_input = soup_form.find('input', {'name': '_csrf'})
    quickTipToken_input = soup_form.find('input', {'name': 'quickTipToken'})

    if not _csrf_input or not quickTipToken_input:
        print("Error: Could not find CSRF token or QuickTip token.")
        return "找不到 CSRF 或 QuickTip Token"

    _csrf = _csrf_input.get("value")
    quickTipToken = quickTipToken_input.get("value")


    train_data = {
        '_csrf': _csrf,
        'custIdTypeEnum': 'PERSON_ID',
        'pid': self.config.person_id,
        'startStation': self.config.from_station,
        'endStation': self.config.to_station,
        'tripType': 'ONEWAY', # 可考慮加入 BookingConfig
        'orderType': 'BY_TRAIN_NO', # 可考慮加入 BookingConfig
        'normalQty': self.config.ticket_num,
        'wheelChairQty': 0,
        'parentChildQty': 0,
        'ticketOrderParamList[0].tripNo': 'TRIP1',
        'ticketOrderParamList[0].rideDate': self.config.getin_date,
        'ticketOrderParamList[0].trainNoList[0]': self.config.train_no[0] if len(self.config.train_no) > 0 else "",
        'ticketOrderParamList[0].trainNoList[1]': self.config.train_no[1] if len(self.config.train_no) > 1 else "",
        'ticketOrderParamList[0].trainNoList[2]': self.config.train_no[2] if len(self.config.train_no) > 2 else "",
        'ticketOrderParamList[0].seatPref': 'NONE', # 可考慮加入 BookingConfig
        # 'ticketOrderParamList[0].chgSeat': 'true',
      ' _ticketOrderParamList[0].chgSeat': 'on',
        'g-recaptcha-response': '',
        'hiddenRecaptcha': '',
        'verifyType': 'voice',
        'verifyCode': verifyCode,
        'quickTipToken': quickTipToken
    }

    if self.config.can_change_seat:
      train_data['ticketOrderParamList[0].chgSeat'] = 'true'

    r = self.session.post('https://www.railway.gov.tw/tra-tip-web/tip/tip001/tip121/bookingTicket', data=train_data)
    soup = BeautifulSoup(r.text, "lxml")
    if soup.find('div', {'id': 'errorDiv'}):
      return (soup.find('div', {'id': 'errorDiv'}).text)
    else:
      return (soup.find('form', {'id': 'order'}).find('strong').text)

  def main(self):
    buy_status = ""
    times = 0
    while('訂票成功' not in buy_status and not self.kill_status):
      times += 1
      # try:
      buy_status = self.buy_tickets()
      if "驗證碼驗證失敗" in buy_status:
        times -= 1
      else:
        # 確保 train_no 列表至少有三個元素，不足則補空字串
        train_no_list_main = list(self.config.train_no) # 創建副本以修改
        while len(train_no_list_main) < 3:
            train_no_list_main.append("")
        
        train_no_0_main = train_no_list_main[0]
        train_no_1_main = train_no_list_main[1]
        train_no_2_main = train_no_list_main[2]

        from_station_name = self.config.from_station.split('-')[1] if '-' in self.config.from_station else self.config.from_station
        to_station_name = self.config.to_station.split('-')[1] if '-' in self.config.to_station else self.config.to_station
        
        res = f"{self.now()}\nt{self.config.getin_date} {from_station_name}👉{to_station_name} {self.config.ticket_num}張 {train_no_0_main} {train_no_1_main} {train_no_2_main}\n第 {times} 次嘗試購票 {buy_status}\n".strip()
        print(res)
      # except:
      #   print("Unexpected error:", sys.exc_info()[0]) # sys 未導入，若要使用需 import sys
        # break

