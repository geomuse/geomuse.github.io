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
string symbol = Symbol() ; 
double lots ; 
int slippage = 10 ; 

// 每日盈亏限制开仓
double DayStartEquity;   // 当日起始权益
int    lastDay;          // 上一次记录的日期
bool   AllowTrade;       // 是否允许新仓

// 退出EA亏损
input double AccumulatedLossLimit = -10000;

// 金字塔加仓
input int step = 50 ;

// 倒金字塔减仓
input double Profit_Threshold     = 100;   // 触发减仓的盈利点数（如100点）
input int    Close_Steps          = 3;     // 减仓次数（如分3次平仓）
input double Close_Percent        = 30;    // 每次平仓比例（如30%）

// 移动止损
input int trailing_start = 130;   // 启动移动止损点数
input int trailing_stop = 70;    // 动态止损点数

// 全仓平仓
input int    TakeProfit = 4000;       // 止盈点数
input int    StopLoss = 200;

// 支撑阻力止损
input double SR_Distance_Points = 50;      // 支撑/阻力位外延点数
input int    SR_Lookback_Bars   = 100;

// 强制平仓持仓超过指定时间的订单
input int    Max_Holding_Minutes = 1200;   // 最大持仓时间（分钟）

// 止损点价格点差
input int StopLossPoint = 200 ;

// SAR
double sar, prev_sar;

// TEMA
int TEMA_Short_Period = 3;    // 短期TEMA周期
int TEMA_Long_Period = 7;     // 长期TEMA周期
int RSI_Period = 7;           // RSI周期
int RSI_Level = 50;            // RSI阈值
int MACD_Fast = 3;            // MACD快线
int MACD_Slow = 7;            // MACD慢线
int MACD_Signal = 9;           // MACD信号线
int Bands_Period = 20;         // 布林带周期
double Bands_Deviation = 2.5;  // 布林带标准差

// 网格
input int      GridStep = 100;          // 网格间距（点）
input int      GridLevels = 5;          // 每方向网格层数

// 随机指标
input int    KPeriod    = 5;       // %K周期
input int    DPeriod    = 3;       // %D周期
input int    Slowing    = 3;       // 平滑参数
input int    Slippage   = 3;       // 允许的滑点
input double LotSize    = 0.1;     // 每次下单手数
input int    Overbought = 80;      // 超买水平
input int    Oversold   = 20;      // 超卖水平

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
    InitializeDailyStats();
    Print("EA初始化完成，起始权益: ", DayStartEquity);
    
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
   
   
   //string symbols[] = {"EURUSD", "GBPUSD", "USDJPY","XAUUSD"}; // 定义需要回测的货币对

  // for(int i=0; i<ArraySize(symbols); i++) {
   // symbol = Symbol() ;
   // magic = generate_magic(symbol) ; 
      
   if(is_friday()) {
     close_all();
     return;
   }
      
   if(TimeCurrent() - lastCheckTime >= CheckInterval)
    {
      CheckMinROI();
      lastCheckTime = TimeCurrent();
    }
      
    // Market();   // enter the markat.
    
    if(balance_check_accumulated_loss()){
      close_all();
      stop_ea();
      return;
    }
   //}
   
       // 检查是否进入新的一天，若是则重置每日统计数据
    if (IsNewDay())
    {
        ResetDailyStats();
    }
    
    // 根据当前权益更新交易权限状态
    UpdateTradePermission();
    
    // 示例：仅当允许新仓时，执行开仓逻辑
    if (AllowTrade)
    {
      Market() ; 
    }
   
  
//---
  }
//+------------------------------------------------------------------+

void Market() {

   TradingStrategy() ;
   stoploss_trailing_stoploss() ; 
   check_intraday_close();     // only trade in day.
    
   //TimeLimitCloser();
   
}

void TradingStrategy(){

   double adx = iADX(Symbol(), 0, 14, PRICE_CLOSE, MODE_MAIN, 0);
   if(adx >= 50 && OrdersTotal()==0){ // 达到50则意味着趋势非常强劲，这可能会成为后续交易决策的条件。
      //close_all() ; 
      // TEMATradingStrategy() ;
   }
   if(adx < 25 && OrdersTotal()==0){
   //   //close_all() ; 
   //   SARTradingStrategy();
      StochasticTradingStrategy() ; 
   }
   if(OrdersTotal()>0){
      TPyramidPosition() ;
   }
   StochasticTradingStrategy() ;
    

}

// int generate_magic(string symbols){
//   magic = 100 ; 
//   if(symbols == "EURUSD") magic = 102 ; 
//   if(symbols == "XAUUSD") magic = 101 ;
//   if(symbols == "GBPUSD") magic = 103 ;
//   if(symbols == "USDJPY") magic = 104 ;  
//   return magic ;
//}

void StochasticTradingStrategy(){
   // 获取当前和前一根K线的随机指标数值
   double currentMain   = iStochastic(NULL, 0, KPeriod, DPeriod, Slowing, MODE_SMA, 0, MODE_MAIN, 0);
   double currentSignal = iStochastic(NULL, 0, KPeriod, DPeriod, Slowing, MODE_SMA, 0, MODE_SIGNAL, 0);
   double prevMain      = iStochastic(NULL, 0, KPeriod, DPeriod, Slowing, MODE_SMA, 0, MODE_MAIN, 1);
   double prevSignal    = iStochastic(NULL, 0, KPeriod, DPeriod, Slowing, MODE_SMA, 0, MODE_SIGNAL, 1);
   
   // 输出调试信息
   Print("当前随机指标主线: ", currentMain, " 信号线: ", currentSignal);
   
   // 多单开仓条件：前一根K线主线低于信号线，当前主线上穿信号线且指标处于超卖区
   if (prevMain < prevSignal && currentMain > currentSignal && currentMain < Oversold && OrdersTotal() == 0 ) {
      // 确保没有已有多单
      lots = 0.01 ; 
      sell() ; 
      }
   
   // 空单开仓条件：前一根K线主线高于信号线，当前主线下穿信号线且指标处于超买区
   if (prevMain > prevSignal && currentMain < currentSignal && currentMain > Overbought && OrdersTotal() == 0) {
      // 确保没有已有空单
      lots = 0.01 ; 
      buy() ; 
   }
}


void SARTradingStrategy(){
    sar = iSAR(Symbol(), 0, 0.02, 0.2, 0);        // 当前周期 SAR 值
    prev_sar = iSAR(Symbol(), 0, 0.02, 0.2, 1);  // 前一周期 SAR 值
    double bid = Bid;
    double ask = Ask;
    
    bool BuySignal = (sar < bid && prev_sar > bid) ;
    bool SELLSignal = (sar > bid && prev_sar < bid) ;
    
    // 执行交易
    if(BuySignal && OrdersTotal() == 0) {
        //close_sell() ; 
        lots = 0.01 ;
        buy() ;
    } else if(SELLSignal && OrdersTotal() == 0) {
        lots = 0.01 ;
        //close_buy() ; 
        sell();
    }
}

void TEMATradingStrategy() {
   
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
    bool BuySignal = (temaShort > temaLong && temaShortPrev <= temaLongPrev) &&
                    (rsi > RSI_Level) &&
                    (macdHist > 0) &&
                    (Close[0] > middleBand && Close[1] <= middleBand);
    
    bool SELLSignal = (temaShort < temaLong && temaShortPrev >= temaLongPrev) &&
                     (rsi < RSI_Level) &&
                     (macdHist < 0) &&
                     (Close[0] < middleBand && Close[1] >= middleBand);
    // 执行交易
    if(BuySignal && OrdersTotal() == 0) {
        lots = 0.01 ;
        //sl = Calculate_SR_StopLoss(OP_BUY) ; 
        buy() ;
    } else if(SELLSignal && OrdersTotal() == 0) {
        lots = 0.01 ;
        //sl = Calculate_SR_StopLoss(OP_SELL) ;  
        sell();
    }
}

void PyramidPosition(){
   if (kbar!=Time[0] && OrdersTotal()>0 && OrdersTotal() <= 5){
      double last_price = last_order_price() ;
      double required_step = step * Point ;
      
      // 买家上涨了，就加买
      if(Bid > last_price + required_step){
         lots = CheckAmount() ;
         //sl = Calculate_SR_StopLoss(OP_BUY) ; 
         buy() ;
      }
      // 卖家下跌了，就加卖
      if(Ask < last_price - required_step){
         lots = CheckAmount() ;
         //sl = Calculate_SR_StopLoss(OP_SELL) ; 
         sell() ;
      }
      kbar = Time[0] ;
   }
}

void TPyramidPosition(){
   if (kbar!=Time[0] && OrdersTotal()>0 && OrdersTotal() <= 5){
      double last_price = last_order_price() ;
      double required_step = step * Point ;
      
      // 买家上涨了，就加买
      if(Bid > last_price + required_step){
         lots =  CheckAmount() ;
         //SetStopLossSell() ; 
         sell() ;
         
      }
      // 卖家下跌了，就加卖
      if(Ask < last_price - required_step){
         lots =  CheckAmount() ;
         //SetStopLossBuy() ;
         buy() ;
      }
      kbar = Time[0] ;
   }

}

double CheckAmount(){
   double lot = NormalizeDouble(MathPow(0.7,OrdersTotal()),2) ;
   if (lot == 0) {
      return 0.01;
   }
   return lot ; 
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

//+------------------------------------------------------------------+
//| 移动止损函数                                                   |
//+------------------------------------------------------------------+
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

//+------------------------------------------------------------------+
//| 买单函数                                                   |
//+------------------------------------------------------------------+
void buy(){
   //sl = Ask - StopLossPoint*Point ;
   //if(lots > 0.34){
   //   sl = Calculate_SR_StopLoss(OP_BUY) ; 
      //sl = 0 ;
   //}
        
   //if(lots < 0.1){
   //   sl = 0 ; 
   //} 
   // sl = DynamicStopLossBuy() ; 
   int ticket = OrderSend(symbol, OP_BUY, lots, Ask, slippage, sl, tp, "buy order", magic, 0, clrBlue);
}

//+------------------------------------------------------------------+
//|  卖单函数                                                   |
//+------------------------------------------------------------------+
void sell(){
   //sl = Bid + StopLossPoint*Point ; 
   
   //if(lots > 0.34){
   //   sl = Calculate_SR_StopLoss(OP_SELL) ; 
      //sl = 0 ;
   //}
        
   //if(lots < 0.1){
   //   sl = 0 ; 
   //}
   // sl = DynamicStopLossSell() ; 
   int ticket = OrderSend(symbol, OP_SELL, lots, Bid, slippage, sl, tp, "sell order", magic, 0, clrRed);
}

//+------------------------------------------------------------------+
//|  平仓函数                                                   |
//+------------------------------------------------------------------+
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
            bool closeSuccess = OrderClose(ticket, lot, price, slippage, clrBlue);
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
            bool closeSuccess = OrderClose(ticket, lots, price, slippage, clrRed);
         }
      }
   }
}

//+------------------------------------------------------------------+
//| ROI                                |
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

double Calculate_SR_StopLoss(int orderType)
{
   double support = Find_Support(SR_Lookback_Bars);
   double resistance = Find_Resistance(SR_Lookback_Bars);

   if(orderType == OP_BUY)
      return support - SR_Distance_Points * Point;
   else if(orderType == OP_SELL)
      return resistance + SR_Distance_Points * Point;
   return 0;
}

//--- 辅助函数：寻找支撑位（最近N根K线的最低点）
double Find_Support(int lookback)
{
   return iLow(Symbol(), 0, iLowest(Symbol(), 0, MODE_LOW, lookback, 0));
}

//--- 辅助函数：寻找阻力位（最近N根K线的最高点）
double Find_Resistance(int lookback)
{
   return iHigh(Symbol(), 0, iHighest(Symbol(), 0, MODE_HIGH, lookback, 0));
}

void Execute_Pyramid_Close(int ticket, int steps, double percent)
{
   if(!OrderSelect(ticket, SELECT_BY_TICKET)) return;

   double lotsToClose = NormalizeDouble(OrderLots() * percent / 100, 2);
   if(lotsToClose <= 0) return;

   for(int i=0; i<steps; i++)
   {
      if(OrderClose(ticket, lotsToClose, OrderClosePrice(), 3, clrNONE))
      {
         Print("部分平仓成功：手数 ", lotsToClose);
         break;
      }
   }
}

void reverse_pyramid_close(){

   for(int i=0; i<OrdersTotal(); i++)
   {
     if(OrderSelect(i, SELECT_BY_POS, MODE_TRADES))
     {
       if(OrderSymbol() == Symbol() && OrderType() <= OP_SELL)
         {
         double currentProfit = OrderProfit() + OrderSwap() + OrderCommission();
         if(currentProfit >= Profit_Threshold * Point * OrderLots())
            {
            Execute_Pyramid_Close(OrderTicket(), Close_Steps, Close_Percent);
            }
         }
      }
    }
}

void TimeLimitCloser(){
   for(int i=OrdersTotal()-1; i>=0; i--)
   {
      if(OrderSelect(i, SELECT_BY_POS, MODE_TRADES))
      {
         // 仅处理当前品种的有效持仓订单
         if(OrderSymbol() == Symbol() && (OrderType() == OP_BUY || OrderType() == OP_SELL))
         {
            datetime openTime = OrderOpenTime();          // 订单开仓时间
            datetime currentTime = TimeCurrent();        // 当前服务器时间
            double elapsedMinutes = (currentTime - openTime) / 60.0;

            if(elapsedMinutes >= Max_Holding_Minutes)
            {
               if(OrderSelect(OrderTicket(), SELECT_BY_TICKET))
               {
                  double closePrice = (OrderType() == OP_BUY) ? Bid : Ask;
                  bool closed = OrderClose(OrderTicket(), OrderLots(), closePrice, 3, clrNONE);
               }
            }
         }
      }
   }

}

//+------------------------------------------------------------------+
//| 删除所有挂单函数                                                 |
//+------------------------------------------------------------------+
void DeleteAllPendingOrders()
{
    for(int i=OrdersTotal()-1; i>=0; i--)
    {
        if(OrderSelect(i, SELECT_BY_POS, MODE_TRADES))
        {
            if(OrderSymbol() == Symbol() && 
               OrderMagicNumber() == magic &&
              (OrderType() == OP_BUYLIMIT || OrderType() == OP_SELLLIMIT))
            {
                bool closeSucuess = OrderDelete(OrderTicket());
            }
        }
    }
}
    
//+------------------------------------------------------------------+
//| Function: InitializeDailyStats                                   |
//| Description: 初始化每日起始权益及日期，并允许交易                   |
//+------------------------------------------------------------------+
void InitializeDailyStats()
{
    lastDay = TimeDay(TimeCurrent());
    DayStartEquity = AccountEquity();
    AllowTrade = true;
}

//+------------------------------------------------------------------+
//| Function: IsNewDay                                                 |
//| Description: 检查是否进入新的一天                                    |
//+------------------------------------------------------------------+
bool IsNewDay()
{
    int currentDay = TimeDay(TimeCurrent());
    return (currentDay != lastDay);
}

//+------------------------------------------------------------------+
//| Function: ResetDailyStats                                          |
//| Description: 重置每日起始权益及允许交易标志                           |
//+------------------------------------------------------------------+
void ResetDailyStats()
{
    lastDay = TimeDay(TimeCurrent());
    DayStartEquity = AccountEquity();
    AllowTrade = true;
    Print("新的一天，重置起始权益: ", DayStartEquity);
}

//+------------------------------------------------------------------+
//| Function: UpdateTradePermission                                    |
//| Description: 根据当日盈亏情况更新交易权限                             |
//+------------------------------------------------------------------+
void UpdateTradePermission()
{
    double currentEquity = AccountEquity();
    
    // 当日盈利超过15%
    /*if (currentEquity >= DayStartEquity * 1.15)
    {
         AllowTrade = false;
         Print("当日盈利超过15%，禁止新仓. 当前权益: ", currentEquity);
    }
    */
    // 当日亏损超过10%
    if (currentEquity <= DayStartEquity * 0.90)
    {
         AllowTrade = false;
         Print("loss 10% : ", currentEquity);
    }
}


//+------------------------------------------------------------------+
//| 动态止损函数                             |
//+------------------------------------------------------------------+

double DynamicStopLoss(){
   double atr = iATR(symbol,0,14,0) ; 
   double stoplosspoint = 2.0*atr ; 
   return stoplosspoint ;
}

double DynamicStopLossBuy(){
   double stoplosspoint = DynamicStopLoss() ; 
   sl = Ask - stoplosspoint*100*Point ;
   Print(stoplosspoint) ;
   return sl  ;
   
}

double DynamicStopLossSell(){
   double stoplosspoint = DynamicStopLoss() ; 
   sl =  Bid + stoplosspoint*100*Point ;
   Print(stoplosspoint) ;
   return sl ;
}

int CountOrders(int orderType) {
   int count = 0;
   for (int i = 0; i < OrdersTotal(); i++) {
      if (OrderSelect(i, SELECT_BY_POS, MODE_TRADES)) {
         if(OrderSymbol() == Symbol() && OrderType() == orderType)
            count++;
      }
   }
   return(count);
}