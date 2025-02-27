//+------------------------------------------------------------------+
//|                                                       gridEA.mq4 |
//|                                  Copyright 2025, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"
#property strict

int magic = 101 ;

// string symbols[3] ; 
string symbol ; 
double lots ; 
int slippage = 10 ; 


input int close_profit = 5 ; 
input int close_loss = 10 ;
input int trailing_start = 50;   // 启动移动止损点数
input int trailing_stop = 20;    // 动态止损点数

input int TEMA_Short_Period = 3;    // 短期TEMA周期
input int TEMA_Long_Period = 7;     // 长期TEMA周期
input int RSI_Period = 7;           // RSI周期
input int RSI_Level = 50;            // RSI阈值
input int MACD_Fast = 3;            // MACD快线
input int MACD_Slow = 7;            // MACD慢线
input int MACD_Signal = 9;           // MACD信号线
input int Bands_Period = 20;         // 布林带周期
input double Bands_Deviation = 2.5;  // 布林带标准差

datetime kbar = 0 ;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---
   
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
  
        symbol = Symbol() ; 
        entry_market();
        check_intraday_close();
//---
  }
//+------------------------------------------------------------------+

void entry_logic(){
  
   if(kbar!=Time[0] && OrdersTotal() == 0 ){
      lots = 0.01 ; buy() ; sell() ; kbar=Time[0] ; 
   }  

}

void CheckOpenConditions() {
    
    // 计算指标
    double temaShort = tema(TEMA_Short_Period, 0);
    double temaLong = tema(TEMA_Long_Period, 0);
    double temaShortPrev = tema(TEMA_Short_Period, 1);
    double temaLongPrev = tema(TEMA_Long_Period, 1);
    
    double rsi = iRSI(NULL, 0, RSI_Period, PRICE_CLOSE, 0);
    double macd = iMACD(NULL, 0, MACD_Fast, MACD_Slow, MACD_Signal, PRICE_CLOSE, MODE_MAIN, 0);
    double macdSignal = iMACD(NULL, 0, MACD_Fast, MACD_Slow, MACD_Signal, PRICE_CLOSE, MODE_SIGNAL, 0);
    double macdHist = macd - macdSignal;
    
    double upperBand = iBands(NULL, 0, Bands_Period, Bands_Deviation, 0, PRICE_CLOSE, MODE_UPPER, 0);
    double lowerBand = iBands(NULL, 0, Bands_Period, Bands_Deviation, 0, PRICE_CLOSE, MODE_LOWER, 0);
    double middleBand = iBands(NULL, 0, Bands_Period, Bands_Deviation, 0, PRICE_CLOSE, MODE_MAIN, 0);

    // 生成信号
    bool buySignal = (temaShort > temaLong && temaShortPrev <= temaLongPrev) &&
                    (rsi > RSI_Level) &&
                    (macdHist > 0) &&
                    (Close[0] > middleBand && Close[1] <= middleBand);
    
    bool sellSignal = (temaShort < temaLong && temaShortPrev >= temaLongPrev) &&
                     (rsi < RSI_Level) &&
                     (macdHist < 0) &&
                     (Close[0] < middleBand && Close[1] >= middleBand);

    // 执行交易
    if(buySignal && OrdersTotal() == 0) {
        lots = 0.1 ;
        buy() ;
    } else if(sellSignal && OrdersTotal() == 0) {
        lots = 0.1 ;
        sell();
    }
}

void entry_market() {

   double total_buy_profit  = get_total_profit_buy() ; 
   double total_sell_profit = get_total_profit_sell() ; 
 
   if(is_friday()) {
        close_all();
        return;
   }
    
   // entry_logic() ;
   CheckOpenConditions() ;
   trailing_stoploss() ;
   
   if(total_buy_profit>15){close_buy(); }
   if(total_sell_profit>15){close_sell();}

}

double ema(int timeframe, int shift){
   return iMA(NULL,0,timeframe,0,MODE_EMA,PRICE_CLOSE,shift);
}

double sma(int timeframe,int shift){
   return iMA(NULL,0,timeframe,0,MODE_SMA,PRICE_CLOSE,shift);
}

double tema(int timeframe, int shift) {
    double ema1 = ema(timeframe, shift);
    double ema2 = ema(timeframe, shift+timeframe);
    double ema3 = ema(timeframe, shift+2*timeframe);
    return (3 * ema1) - (3 * ema2) + ema3;
}

bool is_friday() {
    return (TimeDayOfWeek(TimeCurrent()) == 5);
}

void stop_ea(){
   ExpertRemove() ; 
}

// 日内平仓检查
void check_intraday_close() {
    datetime currentTime = TimeCurrent();
    for(int i = OrdersTotal()-1; i >= 0; i--) {
        if(OrderSelect(i, SELECT_BY_POS, MODE_TRADES)) {
            if(OrderSymbol() == Symbol() && OrderMagicNumber() == magic) {
                if(TimeDay(OrderOpenTime()) != TimeDay(currentTime)) {
                    bool closeSuccess = OrderClose(OrderTicket(), OrderLots(), OrderClosePrice(), 3, clrNONE);
                    if(!closeSuccess){
                     Print("check intraday close problem.");
                    }                   
                  }
            }
        }
    }
}

void trailing_stoploss()
{
   for(int i=0; i<OrdersTotal(); i++)
   {
      if(OrderSelect(i, SELECT_BY_POS, MODE_TRADES))
      {
         if(OrderSymbol() == symbol && OrderMagicNumber() == magic)
         {
            double current_profit = 0;
            
            // 计算当前浮动盈亏（点数）
            if(OrderType() == OP_BUY) 
               current_profit = (Bid - OrderOpenPrice()) / Point;
            if(OrderType() == OP_SELL) 
               current_profit = (OrderOpenPrice() - Ask) / Point;
            
            // 当盈利超过启动点数时执行
            if(current_profit >= trailing_start)
            {
               double newStopLoss = 0;
               bool modify = false;
               
               // 计算新止损价
               if(OrderType() == OP_BUY)
               {
                  newStopLoss = Bid - trailing_stop*Point;
                  if(newStopLoss > OrderStopLoss() || OrderStopLoss() == 0)
                     modify = true;
               }
               
               if(OrderType() == OP_SELL)
               {
                  newStopLoss = Ask + trailing_stop*Point;
                  if(newStopLoss < OrderStopLoss() || OrderStopLoss() == 0)
                     modify = true;
               }
               
               // 执行修改
               if(modify)
               {
                  bool res = OrderModify(OrderTicket(), OrderOpenPrice(), 
                           newStopLoss, OrderTakeProfit(), 0, clrNONE);
                  if(!res)
                     Print("trailing stop loss ", GetLastError());
               }
            }
         }
      }
   }
}

void buy(){
   int ticket = OrderSend(symbol, OP_BUY, lots, Ask, slippage, 0, 0, "buy order", magic, 0, clrBlue);
}

void sell(){
   int ticket = OrderSend(symbol, OP_SELL, lots, Bid, slippage, 0, 0, "sell order", magic, 0, clrRed);
}

double get_total_profit_buy() {
   double totalProfit = 0;
   for (int i = 0; i < OrdersTotal(); i++) {
      if (OrderSelect(i, SELECT_BY_POS, MODE_TRADES)) {  
         if (OrderType() == OP_BUY) { 
            totalProfit += OrderProfit() + OrderSwap() + OrderCommission();
         }
      }
   }
   return totalProfit; 
}

double get_total_profit_sell() {
   double totalProfit = 0;
   for (int i = 0; i < OrdersTotal(); i++) {
      if (OrderSelect(i, SELECT_BY_POS, MODE_TRADES)) {  
         if (OrderType() == OP_SELL) { 
            totalProfit += OrderProfit() + OrderSwap() + OrderCommission();
         }
      }
   }
   return totalProfit; 
}



void close_profit_per_order() {
   for (int i = OrdersTotal() - 1; i >= 0; i--) { 
      if (OrderSelect(i, SELECT_BY_POS, MODE_TRADES)) {  
         if (OrderType() == OP_BUY || OrderType() == OP_SELL) {  
            double profit = OrderProfit() + OrderSwap() + OrderCommission(); 
            if (profit >= close_profit) {  
               bool closeSuccess = OrderClose(OrderTicket(), OrderLots(), 
                                  OrderType() == OP_BUY ? Bid : Ask, 3);
            }
         }
      }
   }
}

void close_loss_per_order() {
   for (int i = OrdersTotal() - 1; i >= 0; i--) { 
      if (OrderSelect(i, SELECT_BY_POS, MODE_TRADES)) {  
         if (OrderType() == OP_BUY || OrderType() == OP_SELL) { 
            double profit = OrderProfit() + OrderSwap() + OrderCommission();  
            if (profit <= close_loss) {  
               bool closeSuccess = OrderClose(OrderTicket(), OrderLots(), 
                                  OrderType() == OP_BUY ? Bid : Ask, 3);
            }
         }
      }
   }
}

void close_buy() {
   for (int i = OrdersTotal() - 1; i >= 0; i--) {
      if (OrderSelect(i, SELECT_BY_POS, MODE_TRADES)) {  
         int ticket = OrderTicket();   
         int type = OrderType();       
         double lot = OrderLots();    
         double price = Bid ;  
        
         if (type == OP_BUY) {
            bool closeSuccess = OrderClose(ticket, lot, price, 3, clrRed);
         }
      }
   }
}

void close_sell() {
   for (int i = OrdersTotal() - 1; i >= 0; i--) {
      if (OrderSelect(i, SELECT_BY_POS, MODE_TRADES)) {  
         int ticket = OrderTicket();    
         int type = OrderType();        
         double lot = OrderLots();     
         double price = Ask;  
       
         if (type == OP_SELL) {
            bool closeSuccess = OrderClose(ticket, lot, price, 3, clrBlue);
         }
      }
   }
}

void close_all() {
   for (int i = OrdersTotal() - 1; i >= 0; i--) {
      if (OrderSelect(i, SELECT_BY_POS, MODE_TRADES)) {  
         int ticket = OrderTicket();    
         int type = OrderType();        
         lots = OrderLots();   
         double price = (type == OP_BUY) ? Bid : Ask;  
         
         if (type == OP_BUY || type == OP_SELL) {
            bool closeSuccess = OrderClose(ticket, lots, price, 3, clrRed);
         }
      }
   }
}