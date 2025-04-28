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
double sl = 0 ;
double tp = 0 ;
// string symbols[3] ; 
string symbol ; 
double lots ; 
int slippage = 10 ; 
int close_profit = 5 ; 
int close_loss = 10 ;

// 退出EA亏损
input double AccumulatedLossLimit = -10000;

// 金字塔加仓
input int step = 130 ;

// 移动止损
input int trailing_start = 110;   // 启动移动止损点数
input int trailing_stop = 70;    // 动态止损点数

// 全仓平仓
input int TakeProfit = 400;       // 止盈点数
input int StopLoss = 200;

// stategy002
double sar, prev_sar;

// stategy001
int TEMA_Short_Period = 3;    // 短期TEMA周期
int TEMA_Long_Period = 7;     // 长期TEMA周期
int RSI_Period = 7;           // RSI周期
int RSI_Level = 50;            // RSI阈值
int MACD_Fast = 3;            // MACD快线
int MACD_Slow = 7;            // MACD慢线
int MACD_Signal = 9;           // MACD信号线
int Bands_Period = 20;         // 布林带周期
double Bands_Deviation = 2.5;  // 布林带标准差

datetime kbar = 0 ;

// ROI
input int CheckInterval    = 5;        
input int      Level1_Time      = 120;       // 1
input double   Level1_ROI       = 0.00;      // 1 ROI 持仓≥120分钟：盈利即可平仓（ROI≥0%）
input int      Level2_Time      = 60;        // 2
input double   Level2_ROI       = 0.02;      // 2 ROI 持仓≥60分钟：需要ROI≥2%
input int      Level3_Time      = 30;        // 3
input double   Level3_ROI       = 0.05;      // 3 ROI 持仓≥30分钟：需要ROI≥5%
input int      Level4_Time      = 0;         // 4
input double   Level4_ROI       = 0.10;      // 4 ROI 持仓≥0分钟：需要ROI≥10%

//--- global variable
double roi_levels[4][2];  // [??, ROI] ???????
datetime lastCheckTime;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---
//---

    // initial ROI 
    roi_levels[0][0] = Level1_Time;
    roi_levels[0][1] = Level1_ROI;
    roi_levels[1][0] = Level2_Time;
    roi_levels[1][1] = Level2_ROI;
    roi_levels[2][0] = Level3_Time;
    roi_levels[2][1] = Level3_ROI;
    roi_levels[3][0] = Level4_Time;
    roi_levels[3][1] = Level4_ROI;
    
    // ???????
    for(int i=0; i<4; i++) {
        if(roi_levels[i][1] < 0) {
            Alert("ROI????????? ", i+1);
            return(INIT_PARAMETERS_INCORRECT);
        }
    }
    
    lastCheckTime = 0;
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
   if(is_friday()) {
        close_all();
        return;
   }
   
   if(TimeCurrent() - lastCheckTime >= CheckInterval)
    {
        CheckMinROI();
        lastCheckTime = TimeCurrent();
    }
  
   symbol = Symbol() ; 
   entry_market();   // enter the markat.
   check_intraday_close();     // only trade in day.
  
   if(balance_check_accumulated_loss()){
      close_all();
      stop_ea();
      return;
   }
//---
  }
//+------------------------------------------------------------------+

void entry_market() {

   entry() ;
   // sar_check_open_condiitons() ;
   // stoploss_trailing_stoploss() ; 
   
}

void entry(){
  
   //double total_buy_profit  = get_total_profit_buy() ; 
   //double total_sell_profit = get_total_profit_sell() ; 
 
   // if(kbar!=Time[0] && OrdersTotal() == 0 ){
   //   lots = 0.1 ; open_position() ; kbar=Time[0] ; 
   // }  
   // sar_check_open_condiitons();
   
   tema_check_open_conditions() ;
   reverse_pyramid_position() ;
  
   // if(total_buy_profit>15){close_buy(); }
   // if(total_sell_profit>15){close_sell();}

   //if(total_buy_profit<-10000){stop_ea() ; }
   //if(total_sell_profit<-10000){stop_ea();}

}

void sar_check_open_condiitons(){
    sar = iSAR(Symbol(), 0, 0.02, 0.2, 0);        // 当前周期 SAR 值
    prev_sar = iSAR(Symbol(), 0, 0.02, 0.2, 1);  // 前一周期 SAR 值
    double bid = Bid;
    double ask = Ask;
    
    bool buy_signal = (sar < bid && prev_sar > bid) ;
    bool sell_signal = (sar > bid && prev_sar < bid) ;
    
    // 执行交易
    if(buy_signal && OrdersTotal() == 0) {
        lots = 0.1 ;
        buy() ;
    } else if(sell_signal && OrdersTotal() == 0) {
        lots = 0.1 ;
        sell();
    }
}

void tema_check_open_conditions() {
   
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
    bool buy_signal = (temaShort > temaLong && temaShortPrev <= temaLongPrev) &&
                    (rsi > RSI_Level) &&
                    (macdHist > 0) &&
                    (Close[0] > middleBand && Close[1] <= middleBand);
    
    bool sell_signal = (temaShort < temaLong && temaShortPrev >= temaLongPrev) &&
                     (rsi < RSI_Level) &&
                     (macdHist < 0) &&
                     (Close[0] < middleBand && Close[1] >= middleBand);
    // 执行交易
    if(buy_signal && OrdersTotal() == 0) {
        lots = 0.1 ;
        buy() ;
    } else if(sell_signal && OrdersTotal() == 0) {
        lots = 0.1 ;
        sell();
    }
}

void pyramid_position(){
   if (kbar!=Time[0] && OrdersTotal()>0 && OrdersTotal() <= 5){
      double last_price = last_order_price() ;
      double required_step = step * Point ;
      
      // 买家上涨了，就加买
      if(Bid > last_price + required_step){
         lots = MathPow(0.7,OrdersTotal()) ;
         buy() ;
      }
      // 卖家下跌了，就加卖
      if(Ask < last_price - required_step){
         lots = MathPow(0.7,OrdersTotal()) ;
         sell() ;
      }
      kbar = Time[0] ;
   }
}

void reverse_pyramid_position(){
   if (kbar!=Time[0] && OrdersTotal()>0 && OrdersTotal() <= 5){
      double last_price = last_order_price() ;
      double required_step = step * Point ;
      
      // 买家上涨了，就加买
      if(Bid > last_price + required_step){
         lots = MathPow(0.7,OrdersTotal()) ;
         sell() ;
      }
      // 卖家下跌了，就加卖
      if(Ask < last_price - required_step){
         lots = MathPow(0.7,OrdersTotal()) ;
         buy() ;
      }
      kbar = Time[0] ;
   }

}



double last_order_price()
{
   double price = 0;
   datetime last_time = 0; // 记录最新订单的时间

   for(int i = 0; i < OrdersTotal(); i++)
   {
      if(OrderSelect(i, SELECT_BY_POS, MODE_TRADES)) // 确保订单选择成功
      {
         if(OrderMagicNumber() == magic && OrderOpenTime() > last_time) 
         {
            // last_time = OrderOpenTime(); // 记录最新的开仓时间
            price = OrderOpenPrice();   // 更新最新订单的开仓价格
         }
      }
   }
   return price;
}

bool open_condition(){
   return (Close[1]>Open[1]) ;
}

void open_position(){
   int type = (open_condition())? OP_BUY : OP_SELL ;
   if(type == OP_BUY){
      buy() ;
   }
   if(type == OP_SELL){
      sell() ;
   }
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
   SendNotification("交易暂停：亏损");
}

bool balance_check_accumulated_loss()
{
    double totalLoss = 0.0;

    // 检查历史订单
    int totalHistoryOrders = OrdersHistoryTotal();
    for(int i=0; i<totalHistoryOrders; i++)
    {
        if(OrderSelect(i, SELECT_BY_POS, MODE_HISTORY))
        {
            if(OrderMagicNumber() == magic && 
               OrderSymbol() == symbol)
            {
               
                double profit = OrderProfit() +
                              OrderSwap() +
                              OrderCommission();
                if (profit < 0) {
                  totalLoss += profit; // 损失为负收益
                }
            }
        }
    }

    // 检查当前持仓（未平仓亏损）
    for(int i=0; i<OrdersTotal(); i++)
    {
        if(OrderSelect(i, SELECT_BY_POS, MODE_TRADES))
        {
            if(OrderMagicNumber() == magic && 
               OrderSymbol() == symbol)
            {
               double profit = OrderProfit() + OrderSwap() + OrderCommission();
               if (profit < 0){
                  totalLoss += profit;
                }
            }
        }
    }

    // 判断是否超过限额
    if(totalLoss <= AccumulatedLossLimit)
    {
        Print("触发累计止损！总损失：", totalLoss);
        return true;
    }
    return false;
}

//+------------------------------------------------------------------+
//| 日内平仓检查                                                   |
//+------------------------------------------------------------------+
void check_intraday_close() {
    datetime currentTime = TimeCurrent();
    for(int i = OrdersTotal()-1; i >= 0; i--) {
        if(OrderSelect(i, SELECT_BY_POS, MODE_TRADES)) {
            if(OrderSymbol() == Symbol() && OrderMagicNumber() == magic) {
                if(TimeDay(OrderOpenTime()) != TimeDay(currentTime)) {
                    bool closeSuccess = OrderClose(OrderTicket(), OrderLots(), OrderClosePrice(), slippage, clrNONE);
                    if(!closeSuccess){
                     Print("check intraday close problem.");
                    }                   
                  }
            }
        }
    }
}

//+------------------------------------------------------------------+
//| 平仓管理函数                                                   |
//+------------------------------------------------------------------+
void stoploss_manage_exit_conditions()
{
   double totalProfit = 0;
   double avgPrice = 0;
   double totalLots = 0;
   
   // 计算整体持仓
   for(int i=0; i<OrdersTotal(); i++)
   {
      if(OrderSelect(i, SELECT_BY_POS) && OrderMagicNumber() == magic)
      {
         totalProfit += OrderProfit();
         totalLots += OrderLots();
         avgPrice += OrderOpenPrice() * OrderLots();
      }
   }
   
   if(totalLots > 0) avgPrice /= totalLots;
   
   // 整体止盈
   if(totalProfit >= TakeProfit * _Point * totalLots * MarketInfo(_Symbol, MODE_TICKVALUE))
   {
      close_all();
      return;
   }
   
   // 整体止损
   double currentSL = (avgPrice > 0) ? 
                     avgPrice - StopLoss * _Point : 
                     avgPrice + StopLoss * _Point;
                     
   if((avgPrice > 0 && Bid <= currentSL) || (avgPrice < 0 && Ask >= currentSL))
   {
      close_all();
   }
}


void stoploss_trailing_stoploss()
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
   int ticket = OrderSend(symbol, OP_BUY, lots, Ask, slippage, sl, tp, "buy order", magic, 0, clrBlue);
}

void sell(){
   int ticket = OrderSend(symbol, OP_SELL, lots, Bid, slippage, sl, tp, "sell order", magic, 0, clrRed);
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

//+------------------------------------------------------------------+
//| ?????????(??????)                                |
//+------------------------------------------------------------------+
double CalculateROI()
{
    double openPrice = OrderOpenPrice();
    double currentPrice;
    
    if(OrderType() == OP_BUY) {
        currentPrice = Bid;
    }
    else if(OrderType() == OP_SELL) {
        currentPrice = Ask;
    }
    else {
        return(0.0); // ?????
    }
    
    double priceDiff = currentPrice - openPrice;
    if(OrderType() == OP_SELL) priceDiff = -priceDiff;
    
    return(NormalizeDouble(priceDiff / openPrice, 4));
}

//+------------------------------------------------------------------+
//| close order                                                    |
//+------------------------------------------------------------------+
bool ExecuteClose()
{
    bool result = false;
    double price;
    
    if(OrderType() == OP_BUY) {
        price = Bid;
    }
    else if(OrderType() == OP_SELL) {
        price = Ask;
    }
    else {
        return(false);
    }
    
    for(int attempt=0; attempt<3; attempt++) { // ????3?
        result = OrderClose(OrderTicket(), OrderLots(), price, 3, clrNONE);
        if(result) break;
        Sleep(500); // ??500ms???
    }
    
    if(!result) {
        Print("????????:", OrderTicket(), " ??:", GetLastError());
    }
    return(result);
}

//+------------------------------------------------------------------+
//| min_roi                                         |
//+------------------------------------------------------------------+
void CheckMinROI()
{
    for(int i=OrdersTotal()-1; i>=0; i--)
    {
        if(!OrderSelect(i, SELECT_BY_POS, MODE_TRADES)) continue;
        
        // ???????????
        if(OrderMagicNumber() != magic) continue;
        
        // ????????(??)
        double holdTimeMinutes = (TimeCurrent() - OrderOpenTime()) / 60.0;
        double currentROI = CalculateROI();
        
        // ?? ROI ??(????)
        for(int j=0; j<4; j++)
        {
            if(holdTimeMinutes >= roi_levels[j][0])
            {
                if(currentROI >= roi_levels[j][1])
                {
                    if(ExecuteClose())
                    {
                        PrintFormat("?? %d ??: ?? %.1f ??, ROI=%.2f%%, ???? %d (?%d??)",
                                    OrderTicket(), holdTimeMinutes, currentROI*100, j+1, roi_levels[j][0]);
                    }
                    break; // ??????????????
                }
                break; // ROI??????????????????????
            }
        }
    }
}
