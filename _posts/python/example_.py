#%%
s1 = "Hello"
s2 = 'Hello'
s3 = """多行字符串"""

print(s1)
print(s2)
print(s3)
# %%
s = "Python"
print(s[0])     # P
print(s[-1])  
# %%
s = "Python"
print(s[1:4])    # yth
print(s[:3])     # Pyt
print(s[3:])     # hon
print(s[::-1])   # n o h t y P (倒序)
# %%
s = "  Python  "
print(s.strip())      # 去掉左右空白
print(s.lstrip())     # 左边空白
print(s.rstrip())     # 右边空白
# %%
s = "Python"
print(s.replace("Python", "Java"))
# %%
s = "Python"
print(s.split("o"))
# %%
print(",".join(["a", "b", "c"]))
# %%
s.endswith(".txt")
# %%
s.startswith("He")

# %%
