import asyncio
from booking_logic import (
    Ticket,
    Mode,
    process_booking
)

async def main():
    example_config = Ticket(
        pid="J123136682",
        start="1000-臺北",
        end="3300-臺中",
        date="2025/05/08",
        mode=Mode.time,
        train=[],
        train_type=[True, True, False],
        start_time="17:00",
        end_time="23:00",
    )
    print(f"準備開始處理 {example_config.pid} 的訂票請求...")
    await process_booking(example_config)
    print(f"{example_config.pid} 的訂票請求處理完畢。請查看上方日誌以了解結果。")


if __name__ == "__main__":
    asyncio.run(main())
