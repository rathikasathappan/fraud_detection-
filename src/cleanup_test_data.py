"""Remove test-source merchant rows and their dependent records from MySQL."""

from src.db import delete_test_merchants


def main() -> None:
    deleted_count = delete_test_merchants()
    print(f"Deleted {deleted_count} test merchant record(s) and related data.")


if __name__ == "__main__":
    main()