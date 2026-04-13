from src.broker.redis_broker import RedisBroker


def main():
    broker = RedisBroker()
    connected = broker.client.ping()
    print("Redis connected:", connected)


if __name__ == "__main__":
    main()
