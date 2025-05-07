import asyncio
import aiohttp
from booking_logic import (
    BookingConfig,
    process_booking
)

async def main():
    example_config = BookingConfig(
        person_id="J123136682",
        from_station="1000-臺北",
        to_station="3300-臺中",
        getin_date="2025/05/09",
        order_type="BY_TIME",
        train_no=["111", "123"],
        ticket_num=1,
        can_change_seat=True,
    )
    print(f"準備開始處理 {example_config.person_id} 的訂票請求...")
    await process_booking(example_config)
    print(f"{example_config.person_id} 的訂票請求處理完畢。請查看上方日誌以了解結果。")


if __name__ == "__main__":
    asyncio.run(main())
