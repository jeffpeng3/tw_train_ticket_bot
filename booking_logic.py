import aiohttp
from asyncio import sleep
from bs4 import BeautifulSoup, Tag
from captcha_handler import CaptchaResolver
from dataclasses import dataclass
from typing import Any
from datetime import datetime, timedelta
from time import time
from enum import Enum


class Mode(Enum):
    ticket = "BY_TRAIN_NO"
    time = "BY_TIME"


@dataclass
class Ticket:
    pid: str
    date: str
    start: str
    end: str
    mode: Mode
    start_time: str
    end_time: str
    train_type: list[bool]
    train: list[str]

    def __post_init__(self):
        if self.mode == Mode.ticket:
            assert not self.start_time
            assert not self.end_time
            assert not self.train_type
            assert self.train
        elif self.mode == Mode.time:
            assert self.start_time
            assert self.end_time
            assert self.train_type
            assert not self.train
        else:
            raise ValueError("Invalid mode")

    def to_payload(
        self, csrf_token: str, quick_tip_token: str, captcha_code: str
    ) -> list[tuple[str, str]]:
        """
        Generates the payload for the booking request.
        The train_data part of the payload is converted to list[tuple[str, str]].
        """

        payload_dict: dict[str, Any] = {
            "_csrf": csrf_token,
            "custIdTypeEnum": "PERSON_ID",
            "pid": self.pid,
            "startStation": self.start,
            "endStation": self.end,
            "tripType": "ONEWAY",
            "normalQty": 1,
            "wheelChairQty": 0,
            "parentChildQty": 0,
            "ticketOrderParamList[0].tripNo": "TRIP1",
            "ticketOrderParamList[0].chgSeat": "true",
            "_ticketOrderParamList[0].chgSeat": "on",
            "verifyType": "voice",
            "verifyCode": captcha_code,
            "g-recaptcha-response": "",
            "hiddenRecaptcha": "",
            "quickTipToken": quick_tip_token,
            "ticketOrderParamList[0].rideDate": self.date,
            "orderType": self.mode.value,
        }
        if self.mode == Mode.ticket:
            for i in range(3):
                if i < len(self.train):
                    payload_dict[f"ticketOrderParamList[0].ticketTypeList[{i}]"] = (
                        self.train[i]
                    )
                else:
                    payload_dict[f"ticketOrderParamList[0].ticketTypeList[{i}]"] = ""
        elif self.mode == Mode.time:
            payload_dict["ticketOrderParamList[0].startOrEndTime"] = (
                "true"  # This seems to be required for BY_TIME
            )
            payload_dict["ticketOrderParamList[0].startTime"] = self.start_time
            payload_dict["ticketOrderParamList[0].endTime"] = self.end_time

        final_payload: list[tuple[str, str]] = []
        for key, value in payload_dict.items():
            if isinstance(value, list):
                for item in value:
                    final_payload.append((key, str(item)))
            else:
                final_payload.append((key, str(value)))

        if self.mode == Mode.time:
            if self.train_type[0]:
                final_payload.append(("ticketOrderParamList[0].trainTypeList", "11"))
            elif self.train_type[1]:
                final_payload.append(("ticketOrderParamList[0].trainTypeList", "2"))
            elif self.train_type[2]:
                final_payload.append(("ticketOrderParamList[0].trainTypeList", "3"))
            for i in range(6):
                final_payload.append(("_ticketOrderParamList[0].trainTypeList", "on"))

        return final_payload


async def _get_initial_page_and_tokens(
    session: aiohttp.ClientSession,
) -> tuple[str, str]:
    async with session.get(
        "https://www.railway.gov.tw/tra-tip-web/tip/tip001/tip121/query"
    ) as response:
        response.raise_for_status()
        html_doc_content: str = await response.text()

    soup: BeautifulSoup = BeautifulSoup(html_doc_content, "lxml")
    soup_form: Tag = soup.find("form")
    _csrf_input: Tag = soup_form.find("input", {"name": "_csrf"})
    quickTipToken_input: Tag = soup_form.find("input", {"name": "quickTipToken"})

    _csrf: str = _csrf_input["value"]
    quickTipToken: str = quickTipToken_input["value"]
    return _csrf, quickTipToken


async def _resolve_captcha(
    session: aiohttp.ClientSession, captcha_resolver: CaptchaResolver
) -> str:
    async with session.get(
        "https://www.railway.gov.tw/tra-tip-web/tip/player/nonPicture?pageRandom=123"
    ) as non_pic_response:
        non_pic_response.raise_for_status()
        _: str = await non_pic_response.text()
        print(_)

    url: str = (
        "https://www.railway.gov.tw/tra-tip-web/tip/player/audio?pageRandom=1079330974"
    )
    verifyCode: str = await captcha_resolver.resolve_audio_captcha(session, url)
    return verifyCode


async def _submit_booking_request(
    session: aiohttp.ClientSession, booking_data: list[tuple[str, str]]
) -> bool:
    async with session.post(
        "https://www.railway.gov.tw/tra-tip-web/tip/tip001/tip121/bookingTicket",
        data=booking_data,
    ) as r:
        r.raise_for_status()
        text_content: str = await r.text()

    soup: BeautifulSoup = BeautifulSoup(text_content, "lxml")
    error_div: Tag | None = soup.find("div", {"id": "errorDiv"})
    return not error_div


async def process_booking(config: Ticket) -> None:
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36"
    headers: dict[str, str] = {"User-Agent": user_agent}
    captcha_resolver_instance = CaptchaResolver()
    endTime: datetime = datetime.now() + timedelta(days=1)
    async with aiohttp.ClientSession(headers=headers) as session:
        while endTime > datetime.now():
            t1: float = time()
            csrf, quick_tip = await _get_initial_page_and_tokens(session)
            captcha = await _resolve_captcha(session, captcha_resolver_instance)
            print(captcha)
            payload: list[tuple[str, str]] = config.to_payload(csrf, quick_tip, captcha)
            if await _submit_booking_request(session, payload):
                print("訂票成功！")
                break
            t2: float = time()
            print(f"使用了 {t2 - t1:.3f} 秒來處理這次請求。")
            await sleep(10)
