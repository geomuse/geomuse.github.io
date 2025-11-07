dplyr basic

filter 挑選 row
arrange 重新安排資料順序 columns 
select 以名稱挑選變數 columns 
mutate 建立新變數 用既有變數的函式運算值建立
summarize 從許多值產生單一的摘要資訊

> diamonds %>% 
+   arrange(color)

> diamonds %>% 
+   arrange(color,cut)
+   arrange(color,cut,depth)
由上面的結果 告訴我們 如果提供一個以上的欄名 ， 前面的資料欄的值平手時決定 新的資料欄  ( 代表優先處理前面的 再處理後面的 )
   
  arrage(.., desc() )  desc >> 遞減的順序  ，  warning : NA 都是在最後
  
> diamonds %>% 
+   select(cut:x)   # cut :: x 之間我都要
+   select(-(cut:x))  # 之外
> diamonds %>% 
+   select(starts_with("c"))
 
 # ends_with("xyz") 結尾以 xyz
 # contains("ijk") 其中含有 ijk
 # num_range("x" , 1:3)  >> x1 , x2 , x3
 # 還有正規式 ， 不過正規式比較難的部分 ， 我會在 Web crawler 最後才補充 
 
 常用的
 
 diamonds %>% 
  select(starts_with("c")) %>% 
    select(color,everything())  #  color 放到最前面 col == 1 
    
  # A tibble: 53,940 x 4
   color carat cut       clarity
   <ord> <dbl> <ord>     <ord>  
 1 E     0.23  Ideal     SI2    
 2 E     0.21  Premium   SI1    
 3 E     0.23  Good      VS1     
 
 > diamonds %>% 
+   select(carat:depth,starts_with("t"),price) %>% 
+     transmute(   # transmute 只呈現新變數
+         x = price/table ,
+         y = (price * depth)/100
+       )

# A tibble: 53,940 x 2
       x     y
   <dbl> <dbl>
 1  5.93  200.
 2  5.34  195.
 3  5.03  186.
 
 > diamonds %>% 
+   select(carat:depth,starts_with("t"),price) %>% 
+     mutate(
+       x = price/table ,
+       y = (price * depth)/100
+     ) %>% 
+   head
# A tibble: 6 x 9
  carat cut       color clarity depth table price     x     y
  <dbl> <ord>     <ord> <ord>   <dbl> <dbl> <int> <dbl> <dbl>
1 0.23  Ideal     E     SI2      61.5    55   326  5.93  200.
2 0.21  Premium   E     SI1      59.8    61   326  5.34  195.
3 0.23  Good      E     VS1      56.9    65   327  5.03  186.

x %>% 
  group_by(group) %>% 
    summarise(count=n(),sum=sum(exhibition),mean=mean(exhibition),var=var(exhibition))
    
 # A tibble: 3 x 5
  group   count   sum  mean     var
  <chr>   <int> <dbl> <dbl>   <dbl>
1 america     6  1015  169.  47010.
2 asia       14 12129  866. 791481.
3 europe     13  2542  196.  34967.