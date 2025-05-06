# -*- coding: utf-8 -*-
import aiohttp
import asyncio
from bs4 import BeautifulSoup, Tag
from datetime import datetime, timezone, timedelta
import captcha_handler
from dataclasses import dataclass, field
from typing import Optional, Any, Coroutine


@dataclass
class BookingConfig:
    person_id: str
    from_station: str
    to_station: str
    getin_date: str
    train_no: list[str] = field(default_factory=lambda: ["", "", ""])
    ticket_num: int
    can_change_seat: bool = True
    normal_ticket_price: Optional[float] = None
    round_trip: bool = False
    return_date: Optional[str] = None
    multi_ride_filter: bool = False
    multi_ride_ticket_type: Optional[str] = None
    student_ticket_type: Optional[str] = None


class RailwayBot():
    # 建構式
    def __init__(self, config: BookingConfig):
        self.config = config
        self.aiohttp_session: Optional[aiohttp.ClientSession] = (
            None  # 新增 aiohttp session
        )
        self.user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36"
        self.captcha_solver: captcha_handler.CaptchaResolver = (
            captcha_handler.CaptchaResolver()
        )

        train_no_list: list[str] = list(self.config.train_no)
        while len(train_no_list) < 3:
            train_no_list.append("")

        train_no_0: str = train_no_list[0]
        train_no_1: str = train_no_list[1]
        train_no_2: str = train_no_list[2]

        start_title: str = f"{self.now()}\n{self.config.getin_date} {self.config.from_station}👉{self.config.to_station} {self.config.ticket_num}張 {train_no_0} {train_no_1} {train_no_2}".strip()
        print(start_title)

        self.kill_status: bool = False
        self.loop: Optional[asyncio.AbstractEventLoop] = None


    async def _create_aiohttp_session(self) -> None:
        if self.aiohttp_session is None or self.aiohttp_session.closed:
            headers: dict[str, str] = {"User-Agent": self.user_agent}
            self.aiohttp_session = aiohttp.ClientSession(headers=headers)

    async def _close_aiohttp_session(self) -> None:
        if self.aiohttp_session and not self.aiohttp_session.closed:
            await self.aiohttp_session.close()

    def run(self) -> None:
        print("開始搶票！")
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        try:
            self.loop.run_until_complete(self._create_aiohttp_session())
            buy_status: str = ""
            times: int = 0
            while "訂票成功" not in buy_status and not self.kill_status:
                times += 1
                buy_status_result: str = self.loop.run_until_complete(
                    self.buy_tickets()
                )

                if isinstance(buy_status_result, str):
                    buy_status = buy_status_result
                else:
                    # This case should ideally not happen if buy_tickets is correctly typed
                    print(
                        f"buy_tickets returned unexpected type: {type(buy_status_result)}"
                    )
                    buy_status = "內部錯誤" # type: ignore

                if (
                    "驗證碼驗證失敗" in buy_status
                    or "aiohttp session 錯誤" in buy_status
                    or "辨識驗證碼失敗" in buy_status
                ):
                    times -= 1
                else:
                    train_no_list_main: list[str] = list(self.config.train_no)
                    while len(train_no_list_main) < 3:
                        train_no_list_main.append("")

                    train_no_0_main: str = train_no_list_main[0]
                    train_no_1_main: str = train_no_list_main[1]
                    train_no_2_main: str = train_no_list_main[2]

                    from_station_name: str = (
                        self.config.from_station.split("-")[1]
                        if "-" in self.config.from_station
                        else self.config.from_station
                    )
                    to_station_name: str = (
                        self.config.to_station.split("-")[1]
                        if "-" in self.config.to_station
                        else self.config.to_station
                    )

                    res: str = f"{self.now()}\nt{self.config.getin_date} {from_station_name}👉{to_station_name} {self.config.ticket_num}張 {train_no_0_main} {train_no_1_main} {train_no_2_main}\n第 {times} 次嘗試購票 {buy_status}\n".strip()
                    print(res)
        finally:
            if self.aiohttp_session and self.loop:
                self.loop.run_until_complete(self._close_aiohttp_session())
            if self.loop:
                self.loop.close()
        print("搶票執行緒結束。")

    def kill(self) -> None:
        self.kill_status = True
        print("已停止搶票！")

    def now(self) -> str:
        dt1: datetime = datetime.now().replace(tzinfo=timezone.utc)
        dt2: datetime = dt1.astimezone(timezone(timedelta(hours=8)))
        return dt2.strftime("%Y-%m-%d %H:%M:%S")

    async def _ensure_session_available(self) -> bool:
        """確保 aiohttp session 可用，如果不可用則嘗試創建。"""
        if not self.aiohttp_session or self.aiohttp_session.closed:
            print(
                "Info: aiohttp_session is not available or closed. Attempting to create."
            )
            await self._create_aiohttp_session()
            if not self.aiohttp_session or self.aiohttp_session.closed:
                print("Error: Failed to create aiohttp_session.")
                return False
        return True

    async def _get_initial_page_and_tokens(self) -> tuple[Optional[str], Optional[str], Optional[str]]:
        """獲取初始訂票頁面，並從中提取 CSRF token 和 QuickTip token。"""
        if not self.aiohttp_session: # Should be handled by _ensure_session_available
            return None, None, "aiohttp session 未初始化"
        try:
            response: aiohttp.ClientResponse
            async with self.aiohttp_session.get(
                "https://www.railway.gov.tw/tra-tip-web/tip/tip001/tip121/query"
            ) as response:
                response.raise_for_status()
                html_doc_content: str = await response.text()
        except aiohttp.ClientError as e:
            print(f"Error accessing booking page: {e}")
            return None, None, "訪問訂票頁面失敗"

        soup: BeautifulSoup = BeautifulSoup(html_doc_content, "lxml")
        soup_form: Optional[Tag] = soup.find("form")
        if not soup_form:
            print("Error: Could not find form in booking page.")
            return None, None, "找不到訂票表單"

        _csrf_input: Optional[Tag] = soup_form.find("input", {"name": "_csrf"})
        quickTipToken_input: Optional[Tag] = soup_form.find("input", {"name": "quickTipToken"})

        if not _csrf_input or not _csrf_input.get("value") or \
           not quickTipToken_input or not quickTipToken_input.get("value"):
            print("Error: Could not find CSRF token or QuickTip token.")
            return None, None, "找不到 CSRF 或 QuickTip Token"

        _csrf: str = _csrf_input["value"] # type: ignore
        quickTipToken: str = quickTipToken_input["value"] # type: ignore
        return _csrf, quickTipToken, None

    async def _resolve_captcha(self) -> tuple[Optional[str], Optional[str]]:
        """處理驗證碼的獲取與解析。"""
        if not self.aiohttp_session: # Should be handled by _ensure_session_available
             return None, "aiohttp session 未初始化"
        try:
            non_pic_response: aiohttp.ClientResponse
            async with self.aiohttp_session.get(
                "https://www.railway.gov.tw/tra-tip-web/tip/player/nonPicture?pageRandom=123"
            ) as non_pic_response:
                non_pic_response.raise_for_status()
        except aiohttp.ClientError as e:
            print(f"Error accessing nonPicture URL: {e}")
            return None, "訪問驗證碼前置頁面失敗"

        actual_audio_captcha_url: str = "https://www.railway.gov.tw/tra-tip-web/tip/player/audio?pageRandom=1079330974"
        # resolve_audio_captcha is an async def, so it returns a Coroutine
        # We need to await it to get the string result.
        verifyCode_coro: Coroutine[Any, Any, str] = self.captcha_solver.resolve_audio_captcha(
            self.aiohttp_session, actual_audio_captcha_url
        )
        verifyCode: str = await verifyCode_coro


        if not verifyCode: # verifyCode will be a string, empty if failed
            print("Error: Failed to resolve captcha audio.")
            return None, "辨識驗證碼失敗"

        print(f"辨識出的驗證碼: {verifyCode}")
        return verifyCode, None

    async def _prepare_booking_data(self, csrf_token: str, quick_tip_token: str, captcha_code: str) -> dict[str, Any]:
        """根據提供的 token 和驗證碼準備訂票請求的資料。"""
        train_data: dict[str, Any] = {
            "_csrf": csrf_token,
            "custIdTypeEnum": "PERSON_ID",
            "pid": self.config.person_id,
            "startStation": self.config.from_station,
            "endStation": self.config.to_station,
            "tripType": "ONEWAY",
            "orderType": "BY_TRAIN_NO",
            "normalQty": self.config.ticket_num,
            "wheelChairQty": 0,
            "parentChildQty": 0,
            "ticketOrderParamList[0].tripNo": "TRIP1",
            "ticketOrderParamList[0].rideDate": self.config.getin_date,
            "ticketOrderParamList[0].trainNoList[0]": self.config.train_no[0]
            if len(self.config.train_no) > 0
            else "",
            "ticketOrderParamList[0].trainNoList[1]": self.config.train_no[1]
            if len(self.config.train_no) > 1
            else "",
            "ticketOrderParamList[0].trainNoList[2]": self.config.train_no[2]
            if len(self.config.train_no) > 2
            else "",
            "ticketOrderParamList[0].seatPref": "NONE",
            " _ticketOrderParamList[0].chgSeat": "on", # Note the leading space, might be intentional or a typo
            "g-recaptcha-response": "",
            "hiddenRecaptcha": "",
            "verifyType": "voice",
            "verifyCode": captcha_code,
            "quickTipToken": quick_tip_token,
        }
        if self.config.can_change_seat:
            train_data["ticketOrderParamList[0].chgSeat"] = "true" # Overwrites the one with space if can_change_seat is true
        return train_data

    async def _submit_booking_request(self, booking_data: dict[str, Any]) -> str:
        """提交訂票請求並處理回應。"""
        if not self.aiohttp_session: # Should be handled by _ensure_session_available
            return "aiohttp session 未初始化"
        try:
            r: aiohttp.ClientResponse
            async with self.aiohttp_session.post(
                "https://www.railway.gov.tw/tra-tip-web/tip/tip001/tip121/bookingTicket",
                data=booking_data,
            ) as r:
                r.raise_for_status() # Raises ClientResponseError for 400-599 status
                text_content: str = await r.text()
                soup: BeautifulSoup = BeautifulSoup(text_content, "lxml")
                error_div: Optional[Tag] = soup.find("div", {"id": "errorDiv"})
                if error_div and error_div.text:
                    return error_div.text.strip()
                
                order_form: Optional[Tag] = soup.find("form", {"id": "order"})
                if order_form:
                    success_msg_strong: Optional[Tag] = order_form.find("strong")
                    if success_msg_strong and success_msg_strong.text:
                        return success_msg_strong.text.strip()
                
                print("Warning: Could not find clear success/error message in booking response.")
                if "系統忙碌中，請稍後再試" in text_content:
                    return "系統忙碌中，請稍後再試"
                if "您的操作過於頻繁" in text_content:
                    return "您的操作過於頻繁，請稍後再試"
                return "無法解析訂票結果，但請求已提交"

        except aiohttp.ClientResponseError as e: # More specific exception for HTTP errors
            print(f"HTTP error during booking ticket submission: {e.status} {e.message}")
            # You might want to return specific messages based on e.status
            return f"訂票請求失敗：{e.status}"
        except aiohttp.ClientError as e: # For other client errors like connection issues
            print(f"Client error during booking ticket submission: {e}")
            return "訂票提交過程中發生網路或客戶端錯誤"
        except Exception as e: # Catch any other unexpected errors
            print(f"Unexpected error during booking submission: {e}")
            return "訂票提交過程中發生未知錯誤"


    async def buy_tickets(self) -> str:
        if not await self._ensure_session_available():
            return "aiohttp session 錯誤"

        csrf_token: Optional[str]
        quick_tip_token: Optional[str]
        error_msg: Optional[str]
        
        csrf_token, quick_tip_token, error_msg = await self._get_initial_page_and_tokens()
        if error_msg or csrf_token is None or quick_tip_token is None: # Ensure tokens are not None
            return error_msg or "未能獲取必要 token"

        captcha_code: Optional[str]
        captcha_code, error_msg = await self._resolve_captcha()
        if error_msg or captcha_code is None: # Ensure captcha_code is not None
            return error_msg or "未能解析驗證碼"

        booking_data: dict[str, Any] = await self._prepare_booking_data(csrf_token, quick_tip_token, captcha_code)
        
        booking_result: str = await self._submit_booking_request(booking_data)
        return booking_result


