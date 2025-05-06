# -*- coding: utf-8 -*-
import requests
from pydub import AudioSegment
import librosa
import numpy as np
import os
from bs4 import BeautifulSoup
from datetime import datetime,timezone,timedelta
import tensorflow as tf
import uuid
import threading


classification = ['b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'w', 'x', 'y', '3', '2', '4', '7', '5', '6', '8', '9', 'a']
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class RailwayBot(threading.Thread):


  # 建構式
  def __init__(self, canChangeSeat, pid, startStation, endStation, normalQty, rideDate, trainNo_1, trainNo_2="", trainNo_3=""):
      threading.Thread.__init__(self)
      self.canChangeSeat = canChangeSeat
      self.pid = pid
      self.startStation=startStation
      self.endStation=endStation
      self.normalQty=normalQty
      self.rideDate=rideDate
      self.trainNo_1=trainNo_1
      self.trainNo_2=trainNo_2
      self.trainNo_3=trainNo_3
      self.file_uuid = uuid.uuid4().hex

      start_title = self.now()+'\n'+rideDate+ ' ' +startStation+"👉"+endStation+" "+normalQty+"張 "+" "+trainNo_1+" "+trainNo_2+" "+trainNo_3
      print(start_title)
      f = open('log/log_'+self.file_uuid+'.txt','w', encoding='UTF-8')
      f.write(start_title)
      f.close()
      # self.main()

      self.kill_status = False


  def run(self):
    print("開始搶票！", self.file_uuid)
    self.main()

  def kill(self):
    self.kill_status = True
    print("已停止搶票！", self.file_uuid)




  def now(self):
    dt1 = datetime.utcnow().replace(tzinfo=timezone.utc)
    dt2 = dt1.astimezone(timezone(timedelta(hours=8))) # 轉換時區 -> 東八區
    return (dt2.strftime("%Y-%m-%d %H:%M:%S")) # 將時間轉換為 string

  def wav2mfcc(self, file_path, max_pad_len=35):
      wave, sr = librosa.load(file_path, mono=True, sr=None)
      wave = wave[::3]
      mfcc = librosa.feature.mfcc(y=wave, sr=16000)
      pad_width = max_pad_len - mfcc.shape[1]
      if pad_width<0:
        print(pad_width)
      mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
      return mfcc


  def get_key_part(self, key_part, sTime, audio_file=False):
    if(not(audio_file)):
      audio_file="audio/"+ self.file_uuid+'_all.mp3'
    song = AudioSegment.from_mp3(audio_file)
    startTime = sTime*1000 - 200
    endTime = (sTime + 1) * 1000
    extract = song[startTime:endTime]
    # Saving
    save_file_name =  "audio/"+ self.file_uuid + "_kp_" + str(key_part) +'.wav'
    extract.export( save_file_name, format="wav")
    return save_file_name

  def predict(self, kp_file):
    # 預測(prediction)
    mfcc = self.wav2mfcc(str(kp_file))
    mfcc_reshaped = mfcc.reshape(1, 20, 35, 1)
    # ans = get_labels()[0][np.argmax(model.predict(mfcc_reshaped))]
    interpreter.set_tensor(input_details[0]['index'], mfcc_reshaped)
    interpreter.invoke()
    ans = interpreter.get_tensor(output_details[0]['index'])
    ans = np.squeeze(ans)
    ans = classification[np.where(ans==np.max(ans))[0][0]]
    # print("tiny", ans)
    # ans = classification[np.argmax(ans))]
    # print("labels=", ans)
    os.remove(kp_file)
    return ans

  def get_ans(self):
    ans=""
    for i in range(6):
      kp_file = self.get_key_part(i+1, 17+i*2)
      # print(predict(kp_file))
      ans += self.predict(kp_file)
    os.remove("audio/"+self.file_uuid+'_all.mp3')
    
    
    return (ans)


  def buy_tickets(self, pid, startStation, endStation, normalQty, rideDate, trainNo_1, trainNo_2, trainNo_3, canChangeSeat):
    headers = { "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_4) AppleWebKit/" "537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36" } 
    response = requests.get("https://www.railway.gov.tw/tra-tip-web/tip/tip001/tip121/query", headers=headers)
    cookies = {"T4TIPSESSIONID": response.cookies['T4TIPSESSIONID'] } 
    requests.get("https://www.railway.gov.tw//tra-tip-web/tip/player/nonPicture?pageRandom=123", headers=headers, cookies=cookies).content
    verifyCode_audio = requests.get("https://www.railway.gov.tw/tra-tip-web/tip/player/audio?pageRandom=1079330974", headers=headers, cookies=cookies).content
    f = open("audio/"+self.file_uuid+'_all.mp3', 'wb')
    f.write(verifyCode_audio)
    f.close()

    html_doc = response.text # text 屬性就是 html 檔案
    soup = BeautifulSoup(html_doc, "lxml") # 指定 lxml 作為解析器

    # 資料
    soup_form = soup.find('form')

    verifyCode= self.get_ans()


    _csrf = soup_form.find('input', {'name': '_csrf'}).get("value")
    quickTipToken = soup_form.find('input', {'name': 'quickTipToken'}).get("value")


    train_data = {
        '_csrf': _csrf, 
        'custIdTypeEnum': 'PERSON_ID',
        'pid': pid,
        'startStation': startStation,
        'endStation': endStation,
        'tripType': 'ONEWAY',
        'orderType': 'BY_TRAIN_NO',
        'normalQty': normalQty,
        'wheelChairQty': 0,
        'parentChildQty': 0,
        'ticketOrderParamList[0].tripNo': 'TRIP1',
        'ticketOrderParamList[0].rideDate': rideDate,
        'ticketOrderParamList[0].trainNoList[0]': trainNo_1,
        'ticketOrderParamList[0].trainNoList[1]': trainNo_2,
        'ticketOrderParamList[0].trainNoList[2]': trainNo_3,
        'ticketOrderParamList[0].seatPref': 'NONE',
        # 'ticketOrderParamList[0].chgSeat': 'true',
      ' _ticketOrderParamList[0].chgSeat': 'on',
        'g-recaptcha-response': '',
        'hiddenRecaptcha': '',
        'verifyType': 'voice',
        'verifyCode': verifyCode,
        'quickTipToken': quickTipToken
    }

    if canChangeSeat :
      train_data['ticketOrderParamList[0].chgSeat'] = 'true'

    r = requests.post('https://www.railway.gov.tw/tra-tip-web/tip/tip001/tip121/bookingTicket', data = train_data, headers=headers, cookies=cookies)
    soup = BeautifulSoup(r.text, "lxml")
    if soup.find('div', {'id': 'errorDiv'}):
      return (soup.find('div', {'id': 'errorDiv'}).text)
    else:
      return (soup.find('form', {'id': 'order'}).find('strong').text)

  def main(self):
    # 參數設定
    canChangeSeat = self.canChangeSeat
    pid = self.pid
    startStation=self.startStation
    endStation=self.endStation
    normalQty=self.normalQty
    rideDate=self.rideDate
    trainNo_1=self.trainNo_1
    trainNo_2=self.trainNo_2
    trainNo_3=self.trainNo_3
    buy_status = ""
    times = 0
    while('訂票成功' not in buy_status and not self.kill_status):
      times += 1
      # try:
      buy_status = self.buy_tickets(pid, startStation, endStation, normalQty, rideDate, trainNo_1, trainNo_2, trainNo_3, canChangeSeat)
      if "驗證碼驗證失敗" in buy_status:
        times -= 1
      else:
        res = str(self.now()) + "\nt" + str(rideDate) + " " + str(startStation.split('-')[1]) + "👉" + str(endStation.split('-')[1]) + " " + str(normalQty)+ "張 " + str(trainNo_1)+ " " + str(trainNo_2)+ " " + str(trainNo_3) + "\n"+"第 "+str(times)+" 次嘗試購票 "+ buy_status+"\n"
        # print(res)
        if times % 2 == 1:
          f = open('log/log_'+self.file_uuid+'.txt','w', encoding='UTF-8')
        else:
          f = open('log/log_'+self.file_uuid+'.txt','a', encoding='UTF-8')
        f.write(res)
        f.close()
      # except:
      #   print("Unexpected error:", sys.exc_info()[0])
        # break



# 初始化
dict_rb = dict()
# 取得網址
# from google.colab.output import eval_js
# print(eval_js("google.colab.kernel.proxyPort(8000)"))


import urllib.parse
import json
from flask import Flask, send_from_directory, jsonify, request

app = Flask(__name__)

@app.route("/")
def home():
    return send_from_directory('static', "index.html")



@app.route("/api/buy", methods=["POST", "GET"])
def buy():
    payload = urllib.parse.unquote_plus(request.args.get('ticket'))
    payload = json.loads(payload)
    rb = RailwayBot(
        payload['canChangeSeat'], 
        payload['pid'], 
        payload['startStation'], 
        payload['endStation'], 
        payload['normalQty'], 
        payload['rideDate'], 
        payload['trainNo_1'], 
        payload['trainNo_2'], 
        payload['trainNo_3']
        )
    dict_rb[rb.file_uuid] = rb
    dict_rb[rb.file_uuid].start()
    return jsonify(
          status = "ok",
          data = payload
      )


@app.route("/api/logs")
def get_All_Log():
    import os
    all_log_file = []
    for x in os.popen('ls log/').readlines():
        all_log_file.append("log/"+x.replace('\n', ''))
    tasks = []
    for log in all_log_file:
        f = open(log, 'r', encoding='UTF-8')
        logs_detal = f.read()
        f.close()
        f = open(log, 'r', encoding='UTF-8')
        logs_detal_array = f.readlines()[1].split(" ",1)
        f.close()
        tasks.append(
            {
                'name': logs_detal_array[1],
                'date': logs_detal_array[0], 
                'log': logs_detal,
                'id': log.split('_')[1].split('.')[0],
             }
        )


                    
    return jsonify(
        tasks = tasks
    )

@app.route("/api/stop")
def stop_app():
    id = request.args.get('id')
    try:
      os.remove('log/log_'+ id +'.txt')
    except:
      print("找不到目標文件")

    try:
      os.remove("audio/"+ id +'_all.mp3')
    except:
      print("找不到目標文件")


    try:
      dict_rb[id].kill()
    except:
      print("找不到目標任務")

    
    return jsonify(status = "ok")




# @app.route("/api/start")
# def start_App():
#     from subprocess import PIPE,Popen
#     process = Popen(['python3', 'main.py'],stderr=PIPE,stdout=PIPE) 
#     return jsonify(
#         status = "ok"
#     )



@app.route('/log/<path:path>')
def send_js(path):
    return send_from_directory('log', path)


# app.run(host="0.0.0.0", port=8000)

if __name__ == "__main__":
    os.makedirs("./audio")
    os.makedirs("./log")
    port = int(os.environ.get("PORT", 8000))
    app.run(host='0.0.0.0', port=port)
