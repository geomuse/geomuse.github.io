def func(nums: list[int]) -> dict[str, int]:
    return {str(n): n for n in nums}

print(func([1,2]))
