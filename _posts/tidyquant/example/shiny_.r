# install.packages(c("tidyquant","dplyr","tidyr"))
library(tidyquant)
library(dplyr)
library(tidyr)

# 1) 直接读取 S&P 500 成分（无需 rvest/xml2）
constituents <- read.csv("C:\\Users\\boonh\\Downloads\\gklln\\_posts\\r\\constituents.csv",stringsAsFactors = FALSE)
syms <- unique(trimws(constituents$Symbol))
syms <- syms[!is.na(syms) & syms != "" & syms != "-"]
syms = syms[1:3]
# 2) 最新收盘价（近30天取最新一日）
prices <- tq_get(syms, get = "stock.prices", from = Sys.Date() - 30) %>%
  group_by(symbol) %>%
  filter(date == max(date, na.rm = TRUE)) %>%
  ungroup() %>%
  transmute(Symbol = symbol, Price = close, Date = date)

# 3) 过去12个月派息合计（容错）
safe_get_div_12m <- function(sym) {
  df <- tryCatch(
    tq_get(sym, get = "dividends", from = Sys.Date() - 365),
    error = function(e) data.frame()
  )
  if (!is.data.frame(df) || nrow(df) == 0 || !"dividends" %in% names(df)) {
    return(data.frame(Symbol = sym, Annual_Dividend = 0, Div_Count = 0, stringsAsFactors = FALSE))
  }
  data.frame(
    Symbol = sym,
    Annual_Dividend = sum(ifelse(is.na(df$dividends), 0, df$dividends)),
    Div_Count = nrow(df),
    stringsAsFactors = FALSE
  )
}

# 聚合（无 purrr）
divs <- do.call(dplyr::bind_rows, lapply(syms, safe_get_div_12m))

# 计算收益率并排序
result <- prices %>%
  dplyr::left_join(divs, by = "Symbol") %>%
  dplyr::mutate(
    Annual_Dividend = ifelse(is.na(Annual_Dividend), 0, Annual_Dividend),
    Dividend_Yield = ifelse(Price > 0, Annual_Dividend / Price * 100, 0)
  ) %>%
  dplyr::arrange(dplyr::desc(Dividend_Yield))

head(result, 30)