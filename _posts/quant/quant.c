//+------------------------------------------------------------------+
//|                                          Grid Point Strategy.mq4 |
//|                        Copyright 2021, MetaQuotes Software Corp. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "geo"
#property link      "geo"
#property version   "0.1"
#property strict

//+------------------------------------------------------------------+
//| global variable and setting control                              |
//+------------------------------------------------------------------+

double order ; 
string currency ; 
double max_order = 100 ; 

int MAGIC = 100 ; int ticket ; int slippage ; 
bool warn = false ; 

//+------------------------------------------------------------------+
//|  local variable and custom control                                |
//+------------------------------------------------------------------+

double stop_less_point  , stop_less_price ,  stop_less_price_1 ; 
double stop_profit_point  , stop_profit_price , stop_profit_price_1 ; 
double buy_stop_distance , buy_limit_distance  , sell_stop_distance , sell_limit_distance ; 
double buy_stop_number = 1 , buy_limit_number = 1 , sell_stop_number = 1 , sell_limit_number = 1 ; 

double history_total_order=0 , history_total_profit = 0 , history_order=0 , history_profit=0 ; 
double mbbo = 0 , mbbuy_not_ordered_profito = 0 , msso = 0 , mssell_not_ordered_profito = 0 , bb= 0 , bbuy_not_ordered_profit = 0 ; 
double ss = 0 , ssell_not_ordered_profit = 0 , bbl = 0, bbuy_not_ordered_profit1 = 0 , ssl = 0 , ssell_not_ordered_profit1 = 0 ; 
double ossa = 0 , osla = 0 , obsa=0 , obla = 0 , twbs = 0 , twin = 0 , tlbs = 0 , tloss = 0 ; 
double slots = 0 , mbb=0 , mbbuy_not_ordered_profit = 0 , blots = 0 , mss=0 , mssell_not_ordered_profit = 0 , moss = 0 , mosl = 0 , mobs = 0 ; 
double mobl = 0 , profitmm = 0 , totallots = 0 , sell_not_ordered_total = 0 , sell_not_ordered_profit = 0 , lastpricebuy = 0 , lastpricesell = 0 ; 
double tlotsb = 0 , buy_not_ordered_total=0 , slastlots = 0 , buy_not_ordered_profit=0 , tlots = 0 , oss=0 , osl=0 , obs = 0 , obl=0 , blastlots = 0 ;
  
datetime one_bar = 0 ;

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
//---
   currency = Symbol() ; 
   account_check() ; 
   check_buy_order() ;
   check_sell_order() ;
   double ma1_0 = iMA(currency,0,6,0,MODE_SMA,PRICE_CLOSE,0) ;
   double ma1_1 = iMA(currency,0,6,0,MODE_SMA,PRICE_CLOSE,1) ;
   
   double ma2_0 = iMA(currency,0,25,0,MODE_SMA,PRICE_CLOSE,0) ;  
   double ma2_1 = iMA(currency,0,25,0,MODE_SMA,PRICE_CLOSE,1) ; 
   
   double volatility = (High[0]-High[1])/High[1] ;
   
   bool gold = false ; 
   bool dead = false ; 
   
   if(ma1_0>ma2_0 && ma1_1<ma2_1){gold = true;}
   if(ma1_0<ma2_0 && ma1_1>ma2_1){dead = true;}
   
   if(gold && one_bar!=Time[0] && buy_not_ordered_total == 0 && MathAbs(volatility)>1e-3){
      order = 0.01 ; buy() ; buy_stop_distance = 65 ; buy_stop_number = 10 ;
      buy_stop() ; buy_limit_distance = 65 ; buy_limit_number = 10 ;
      buy_limit() ; one_bar=Time[0] ; 
   } 
   
   if(dead && one_bar!=Time[0] && sell_not_ordered_total == 0 && MathAbs(volatility)>1e-3){
      order = 0.01 ; sell() ; sell_stop_distance = 65 ; sell_stop_number = 10 ;
      sell_stop() ; sell_limit_distance = 65 ; sell_limit_number = 10 ;
      sell_limit() ; one_bar = Time[0] ;
   }
   
   if(buy_not_ordered_profit>30){close_buy() ; close_buystop() ; close_buylimit() ;}
   if(sell_not_ordered_profit>30){close_sell() ; close_sellstop() ; close_selllimit() ;} 
  }
  
//+------------------------------------------------------------------+
void account_check(){
   history_total_order=0 ; history_total_profit = 0 ; history_order=0 ; history_profit=0 ; 
   mbbo = 0 ; mbbuy_not_ordered_profito = 0 ; msso = 0 ; mssell_not_ordered_profito = 0 ; bb= 0 ; bbuy_not_ordered_profit = 0 ; 
   ss = 0 ; ssell_not_ordered_profit = 0 ; bbl = 0; bbuy_not_ordered_profit1 = 0 ; ssl = 0 ; ssell_not_ordered_profit1 = 0 ; 
   ossa = 0 ; osla = 0 ; obsa=0 ; obla = 0 ; twbs = 0 ; twin = 0 ; tlbs = 0 ; tloss = 0 ; 
   slots = 0 ; mbb=0 ; mbbuy_not_ordered_profit = 0 ; blots = 0 ; mss=0 ; mssell_not_ordered_profit = 0 ; moss = 0 ; mosl = 0 ; mobs = 0 ; 
   mobl = 0 ; profitmm = 0 ; totallots = 0 ; sell_not_ordered_total = 0 ; sell_not_ordered_profit = 0 ; lastpricebuy = 0 ; lastpricesell = 0 ; 
   tlotsb = 0 ; buy_not_ordered_total=0 ; slastlots = 0 ; buy_not_ordered_profit=0 ; tlots = 0 ; oss=0 ; osl=0 ; obs = 0 ; obl=0 ; blastlots = 0 ;
   for(int r=0;r<OrdersHistoryTotal() ; r++){
      if(OrderSelect(r,SELECT_BY_POS,MODE_HISTORY)){
         if(OrderType()==OP_BUY || OrderType()==OP_SELL){
            history_total_order+=OrderLots() ; 
            history_total_profit+=OrderProfit()+OrderCommission()+OrderSwap() ; 
         }
         if(OrderSymbol()==currency){
            history_order +=OrderLots() ; 
            history_profit += OrderProfit() + OrderCommission() + OrderSwap() ; 
            if(OrderType()==OP_BUY){
               mbbo++ ; mbbuy_not_ordered_profito += OrderProfit() + OrderSwap() + OrderCommission() ; 
            }
            if(OrderType()==OP_SELL){
               msso++ ; mssell_not_ordered_profito += OrderProfit() + OrderSwap() + OrderCommission() ; 
            }
         }
      }
   }
   for(int cnt=0 ; cnt < OrdersTotal() ; cnt++){
      if(OrderSelect(cnt,SELECT_BY_POS,MODE_TRADES)){
         if(OrderType()==OP_BUY && OrderMagicNumber() == MAGIC){
            bb++ ; bbuy_not_ordered_profit+=OrderProfit() + OrderSwap() + OrderCommission() ; 
         }
         if(OrderType()==OP_SELL && OrderMagicNumber()==MAGIC){
            ss++ ; ssell_not_ordered_profit+=OrderProfit()+OrderSwap()+OrderCommission() ; 
         }
         if(OrderType()==OP_BUY){
            bbl++ ; bbuy_not_ordered_profit1+=OrderProfit()+OrderSwap()+OrderCommission() ;
         }
         if(OrderType()==OP_SELL){
            ssl++ ; ssell_not_ordered_profit1+=OrderProfit()+OrderSwap()+OrderCommission() ; 
         }
         if(OrderType()==OP_SELLSTOP){
            ossa ++ ; 
         }
         if(OrderType()==OP_SELLLIMIT){
            osla ++ ; 
         }
         if(OrderType()==OP_BUYSTOP){
            obsa++ ;
         }
         if(OrderType()==OP_BUYLIMIT){
            obla++ ; 
         }
         if((OrderType()==OP_BUY||OrderType()==OP_SELL)&&(OrderProfit()+OrderSwap()+OrderCommission())>0){
            twbs++ ; twin+=OrderProfit()+OrderSwap()+OrderCommission() ; 
         }
         if((OrderType()==OP_BUY||OrderType()==OP_SELL)&&(OrderProfit()+OrderSwap()+OrderCommission())<0){
            tlbs++ ; tloss+=OrderProfit()+OrderSwap()+OrderCommission(); 
         }
         if(OrderType()==OP_BUY||OrderType()==OP_SELL){
            totallots+=OrderLots() ;
         }
         if(OrderSymbol()==currency){
            if(OrderType()==OP_BUY){
               blots+=OrderLots() ; mbb++ ; 
               mbbuy_not_ordered_profit+=OrderProfit()+OrderSwap()+OrderCommission() ; 
            }
            if(OrderType()==OP_SELL){
               slots+=OrderLots() ; mss++ ; 
               mssell_not_ordered_profit+=OrderProfit()+OrderSwap()+OrderCommission() ;
            }
            if(OrderType()==OP_SELLSTOP){moss++;}
            if(OrderType()==OP_SELLLIMIT){mosl++;}
            if(OrderType()==OP_BUYSTOP){mobs++;}
            if(OrderType()==OP_BUYLIMIT){mobl++;}
            profitmm+=OrderProfit()+OrderSwap()+OrderCommission() ;
         }
        if(OrderSymbol()==currency&&OrderMagicNumber()==MAGIC){
            if(OrderType()==OP_SELL){
               tloss+=OrderLots() ; sell_not_ordered_total++ ; slastlots=OrderLots() ; sell_not_ordered_profit+=OrderProfit()+OrderSwap()+OrderCommission() ;
               lastpricesell = OrderOpenPrice() ; 
            }
            if(OrderType()==OP_BUY){
               tlotsb+=OrderLots() ; buy_not_ordered_total++ ; blastlots=OrderLots() ; 
               buy_not_ordered_profit+=OrderProfit()+OrderSwap()+OrderCommission() ;
               lastpricebuy = OrderOpenPrice() ; 
            }
            if(OrderType()==OP_SELL||OrderType()==OP_BUY){
               tlots+=OrderLots() ; 
            }
            if(OrderType()==OP_SELLSTOP){oss++;}
            if(OrderType()==OP_SELLLIMIT){osl++;}
            if(OrderType()==OP_BUYSTOP){obs++;}
            if(OrderType()==OP_BUYLIMIT){obl++;}
        }
      }
   }
} 

void check_buy_order(){
   for(int i = 0 ; i < OrdersTotal() ; i++){
      if(OrderSelect(i,SELECT_BY_POS,MODE_TRADES)){
         if(OrderType() == OP_BUY){
            double profit = OrderProfit() + OrderSwap() + OrderCommission() ;
            if(profit < -30){
               OrderClose(OrderTicket(), OrderLots(), Ask, 3, Red);
            }
         }
      }
   }
}

void check_sell_order(){
   for(int i = 0 ; i < OrdersTotal() ; i++){
      if(OrderSelect(i,SELECT_BY_POS,MODE_TRADES)){
         if(OrderType() == OP_SELL){
            double profit = OrderProfit() + OrderSwap() + OrderCommission() ;
            if(profit < -30){
               OrderClose(OrderTicket(), OrderLots(), Ask, 3, Red);
            }
         }
      }
   }
}

void buy(){
   order = NormalizeDouble(order,2) ; 
   if(order < MarketInfo(currency,MODE_MINLOT)){
      order = MarketInfo(currency,MODE_MINLOT) ;
   }
   
   if(order > max_order){
      order = max_order ;
   }
   
   if(order > MarketInfo(currency,MODE_MAXLOT)){
      order = MarketInfo(currency,MODE_MAXLOT) ; 
   }
   
   if(stop_profit_point==0){
      stop_profit_price = 0 ; 
   }
   
   if(stop_profit_point>0){
      stop_profit_price = MarketInfo(currency,MODE_ASK) + stop_profit_point*MarketInfo(currency,MODE_POINT)  ; 
   }
   
   if(stop_less_point==0){
      stop_less_price = 0 ;
   }
   
   if(stop_less_point>0){
      stop_less_price = MarketInfo(currency,MODE_ASK) - stop_less_point * MarketInfo(currency,MODE_POINT) ; 
   }
   
   // above is finish all about buy comment 
   
   ticket = OrderSend(currency,OP_BUY,order,MarketInfo(currency,MODE_ASK),ticket,stop_less_price,stop_profit_price,"buy one",MAGIC,0,Violet);
   
   if(ticket<0){
      if(warn){
         Alert("your long orders isn't sucessful",GetLastError()) ; 
      }
   }
   else{
      if(warn){
         Alert("your long orders is sucessfully") ; 
      }
   }
} 

void sell(){
   order = NormalizeDouble(order,2) ; 
   if(order < MarketInfo(currency,MODE_MINLOT)){
      order = MarketInfo(currency,MODE_MINLOT) ;
   }
   
   if(order > max_order){
      order = max_order ;
   }
   
   if(order > MarketInfo(currency,MODE_MAXLOT)){
      order = MarketInfo(currency,MODE_MAXLOT) ; 
   }
   
   if(stop_profit_point==0){
      stop_profit_price = 0 ; 
   }
   
   if(stop_profit_point>0){
      stop_profit_price = MarketInfo(currency,MODE_BID) - stop_profit_point*MarketInfo(currency,MODE_POINT)  ; 
   }
   
   if(stop_less_point==0){
      stop_less_price = 0 ;
   }
   
   if(stop_less_point>0){
      stop_less_price = MarketInfo(currency,MODE_BID) + stop_less_point * MarketInfo(currency,MODE_POINT) ; 
   }
   
   // above is finish all about buy comment 
   
   ticket = OrderSend(currency,OP_SELL,order,MarketInfo(currency,MODE_BID),ticket,stop_less_price,stop_profit_price,"sell one",MAGIC,0,GreenYellow);
   
   if(ticket<0){
      if(warn){
         Alert("your short orders isn't sucessful",GetLastError()) ; 
      }
   }
   else{
      if(warn){
         Alert("your short orders is sucessfully") ; 
      }
   }
}

void close_buy(){
   double sell_price ;
   double hands_short ; 
   int order_type ; 
   int i ; 
   bool result = false ; 
   int order_number ; 
   for(i = OrdersTotal()-1 ; i>=0 ; i--){
      if(OrderSelect(i,SELECT_BY_POS)){
         if(OrdersTotal()>0 && OrderSymbol()==currency&&OrderMagicNumber()==MAGIC){
            sell_price = MarketInfo(currency,MODE_BID) ; 
            order_number = OrderTicket() ; 
            hands_short = OrderLots() ; 
            order_type = OrderType() ;
            switch(order_type){
               // if buy , close it 
               case OP_BUY : result = OrderClose(order_number,hands_short,sell_price,ticket,Yellow) ; 
               break ; 
            }
         }
      }
   }
}

void close_sell(){   
   double buy_price ;
   double hands_short ; 
   int order_type ; 
   int i ; 
   bool result = false ; 
   int order_number ; 
   for(i = OrdersTotal()-1 ; i>=0 ; i--){
      if(OrderSelect(i,SELECT_BY_POS)){
         if(OrdersTotal()>0 && OrderSymbol()==currency&&OrderMagicNumber()==MAGIC){
            buy_price = MarketInfo(currency,MODE_ASK) ; 
            order_number = OrderTicket() ; 
            hands_short = OrderLots() ; 
            order_type = OrderType() ;
            switch(order_type){
               // if buy , close it 
               case OP_SELL : 
                  result = OrderClose(order_number,hands_short,buy_price,ticket,Red) ; 
                  break ; 
            }
         }
      }
   }
}


void close_profit(){
   double buy_price ; 
   double sell_price ; 
   int order_number ; 
   double hands_short ;
   int order_type ; 
   bool result = false ; 
   
   for(int i=OrdersTotal()-1;i>=0;i--){
      if(OrderSelect(i,SELECT_BY_POS)){
         // choose profit order
         if(OrderSymbol()==currency&&OrderProfit()+OrderSwap()+OrderCommission()>0&&OrderMagicNumber()==MAGIC){
            buy_price = MarketInfo(OrderSymbol(),MODE_ASK) ;
            sell_price = MarketInfo(OrderSymbol(),MODE_BID);
            order_number = OrderTicket();
            hands_short = OrderLots() ; 
            order_type = OrderType() ;
            switch(order_type){
               case OP_BUY : 
                  result = OrderClose(order_number,hands_short,sell_price,ticket,Yellow) ; 
                  if(warn){
                     Alert(currency + " buy order(profit) close") ; 
                  }
                  break ;
               case OP_SELL : 
                  result = OrderClose(order_number,hands_short,buy_price,ticket,Red) ; 
                  if(warn){
                     Alert(currency + " sell order(profit) close") ; 
                  }
                  break ; 
            }
            if(result==false){
               if(warn){
                  Alert("EA close proft order is failed") ; 
               }
            }        
         }  
      }
   }
}

void close_less(){
   double buy_price ; 
   double sell_price ; 
   int order_number ; 
   double hands_short ;
   int order_type ; 
   bool result = false ; 
   
   for(int i=OrdersTotal()-1;i>=0;i--){
      if(OrderSelect(i,SELECT_BY_POS)){
         // choose profit order
         if(OrderSymbol()==currency&&OrderProfit()+OrderSwap()+OrderCommission()<0&&OrderMagicNumber()==MAGIC){
            buy_price = MarketInfo(OrderSymbol(),MODE_ASK) ;
            sell_price = MarketInfo(OrderSymbol(),MODE_BID);
            order_number = OrderTicket();
            hands_short = OrderLots() ; 
            order_type = OrderType() ;
            switch(order_type){
               case OP_BUY : 
                  result = OrderClose(order_number,hands_short,sell_price,ticket,Yellow) ; 
                  if(warn){
                     Alert(currency + " buy order(less) close") ; 
                  }
                  break ;
               case OP_SELL : 
                  result = OrderClose(order_number,hands_short,buy_price,ticket,Red) ; 
                  if(warn){
                     Alert(currency + " sell order(less) close") ; 
                  }
                  break ; 
            }
            if(result==false){
               if(warn){
                  Alert("EA close proft order is failed") ; 
               }
            }        
         }  
      }
   }
}

void buy_stop(){
   order = NormalizeDouble(order,2) ; 
   if(order < MarketInfo(currency,MODE_MINLOT)){
      order = MarketInfo(currency,MODE_MINLOT) ;
   }
   if(order > max_order){
      order = max_order ; 
   }
   if(order > MarketInfo(currency,MODE_MAXLOT)){
      order  = MarketInfo(currency,MODE_MAXLOT) ;
   }
   if(buy_stop_distance > 2 && buy_stop_number > 0){
      for(int K=1;K<=buy_stop_number ; K++){
         if(stop_profit_point == 0){
            stop_profit_price_1 = 0 ; stop_profit_price = 0 ; 
         }
         if(stop_profit_point>0){
            stop_profit_price_1 = MarketInfo(currency,MODE_ASK) + (stop_profit_point*MarketInfo(currency,MODE_POINT)) ; 
            stop_profit_price = stop_profit_price_1 + (K*(buy_stop_distance*MarketInfo(currency,MODE_POINT)));
         }
         if(stop_less_point==0){
            stop_less_price_1 = 0 ; stop_less_price = 0 ;
         }
         if(stop_less_point>0){
            stop_less_price_1 = MarketInfo(currency,MODE_ASK) - (stop_less_point*MarketInfo(currency,MODE_POINT)) ;
            stop_less_price = stop_less_price_1 + (K*(buy_stop_distance*MarketInfo(currency,MODE_POINT))); 
         }
         ticket = OrderSend(currency,OP_BUYSTOP,order,MarketInfo(currency,MODE_ASK)+(K*buy_stop_distance*MarketInfo(currency,MODE_POINT)),slippage,stop_less_price,stop_profit_price,"BUYSTOP",MAGIC,0,Green);   
      }
      if(ticket<0){
         if(warn){
            Alert("BUYSTOP failed") ;
         }
      }
      else{
        if(warn){
            Alert("BUYSTOP successfull") ; 
        }
      } 
   }
   else return ; 
}

void buy_limit(){
   order = NormalizeDouble(order,2) ; 
   if(order < MarketInfo(currency,MODE_MINLOT)){
      order = MarketInfo(currency,MODE_MINLOT) ;
   }
   if(order > max_order){
      order = max_order ; 
   }
   if(order > MarketInfo(currency,MODE_MAXLOT)){
      order  = MarketInfo(currency,MODE_MAXLOT) ;
   }
   if(buy_limit_distance > 2 && buy_limit_number > 0){
      for(int K=1;K<=buy_limit_number ; K++){
         if(stop_profit_point == 0){
            stop_profit_price_1 = 0 ; stop_profit_price = 0 ; 
         }
         if(stop_profit_point>0){
            stop_profit_price_1 = MarketInfo(currency,MODE_ASK) + (stop_profit_point*MarketInfo(currency,MODE_POINT)) ; 
            stop_profit_price = stop_profit_price_1 - (K*(buy_limit_distance*MarketInfo(currency,MODE_POINT)));
         }
         if(stop_less_point==0){
            stop_less_price_1 = 0 ; stop_less_price = 0 ;
         }
         if(stop_less_point>0){
            stop_less_price_1 = MarketInfo(currency,MODE_ASK) - (stop_less_point*MarketInfo(currency,MODE_POINT)) ;
            stop_less_price = stop_less_price_1 - (K*(buy_limit_distance*MarketInfo(currency,MODE_POINT))); 
         }
         ticket = OrderSend(currency,OP_BUYLIMIT,order,MarketInfo(currency,MODE_ASK)-(K*buy_limit_distance*MarketInfo(currency,MODE_POINT)),slippage,stop_less_price,stop_profit_price,"BUYLIMIT",MAGIC,0,Green);   
      }
      if(ticket<0){
         if(warn){
            Alert("BUYLIMIT failed") ;
         }
      }
      else{
        if(warn){
            Alert("BUYLIMIT successfull") ; 
        }
      } 
   }
   else return ; 
}

void sell_stop(){
   order = NormalizeDouble(order,2) ; 
   if(order < MarketInfo(currency,MODE_MINLOT)){
      order = MarketInfo(currency,MODE_MINLOT) ;
   }
   if(order > max_order){
      order = max_order ; 
   }
   if(order > MarketInfo(currency,MODE_MAXLOT)){
      order  = MarketInfo(currency,MODE_MAXLOT) ;
   }
   if(sell_stop_distance > 2 && sell_stop_number > 0){
      for(int K=1;K<=sell_stop_number ; K++){
         if(stop_profit_point == 0){
            stop_profit_price_1 = 0 ; stop_profit_price = 0 ; 
         }
         if(stop_profit_point>0){
            stop_profit_price_1 = MarketInfo(currency,MODE_ASK) - (stop_profit_point*MarketInfo(currency,MODE_POINT)) ; 
            stop_profit_price = stop_profit_price_1 - (K*(sell_stop_distance*MarketInfo(currency,MODE_POINT)));
         }
         if(stop_less_point==0){
            stop_less_price_1 = 0 ; stop_less_price = 0 ;
         }
         if(stop_less_point>0){
            stop_less_price_1 = MarketInfo(currency,MODE_ASK) + (stop_less_point*MarketInfo(currency,MODE_POINT)) ;
            stop_less_price = stop_less_price_1 - (K*(sell_stop_distance*MarketInfo(currency,MODE_POINT))); 
         }
         ticket = OrderSend(currency,OP_SELLSTOP,order,MarketInfo(currency,MODE_ASK)-(K*sell_stop_distance*MarketInfo(currency,MODE_POINT)),slippage,stop_less_price,stop_profit_price,"SELLSTOP",MAGIC,0,Green);   
      }
      if(ticket<0){
         if(warn){
            Alert("SELLSTOP failed") ;
         }
      }
      else{
        if(warn){
            Alert("SELLSTOP successfull") ; 
        }
      } 
   }
   else return ; 
}

void sell_limit(){
   order = NormalizeDouble(order,2) ; 
   if(order < MarketInfo(currency,MODE_MINLOT)){
      order = MarketInfo(currency,MODE_MINLOT) ;
   }
   if(order > max_order){
      order = max_order ; 
   }
   if(order > MarketInfo(currency,MODE_MAXLOT)){
      order  = MarketInfo(currency,MODE_MAXLOT) ;
   }
   if(sell_limit_distance > 2 && sell_limit_number > 0){
      for(int K=1;K<=sell_limit_number ; K++){
         if(stop_profit_point == 0){
            stop_profit_price_1 = 0 ; stop_profit_price = 0 ; 
         }
         if(stop_profit_point>0){
            stop_profit_price_1 = MarketInfo(currency,MODE_ASK) - (stop_profit_point*MarketInfo(currency,MODE_POINT)) ; 
            stop_profit_price = stop_profit_price_1 + (K*(sell_limit_distance*MarketInfo(currency,MODE_POINT)));
         }
         if(stop_less_point==0){
            stop_less_price_1 = 0 ; stop_less_price = 0 ;
         }
         if(stop_less_point>0){
            stop_less_price_1 = MarketInfo(currency,MODE_ASK) + (stop_less_point*MarketInfo(currency,MODE_POINT)) ;
            stop_less_price = stop_less_price_1 + (K*(sell_limit_distance*MarketInfo(currency,MODE_POINT))); 
         }
         ticket = OrderSend(currency,OP_SELLLIMIT,order,MarketInfo(currency,MODE_ASK)+(K*sell_limit_distance*MarketInfo(currency,MODE_POINT)),slippage,stop_less_price,stop_profit_price,"SELLSTOP",MAGIC,0,Green);   
      }
      if(ticket<0){
         if(warn){
            Alert("SELLSTOP failed") ;
         }
      }
      else{
        if(warn){
            Alert("SELLSTOP successfull") ; 
        }
      } 
   }
   else return ; 
}

void close_buystop(){
   int order_number = 0 ; 
   int order_type ; 
   int i ; 
   bool result = false ; 
   for(i=OrdersTotal()-1;i>=0;i--){
      if(OrderSelect(i,SELECT_BY_POS)){
         if(OrderSymbol()==currency&&OrderMagicNumber()==MAGIC){
            order_number = OrderTicket() ; 
            order_type = OrderType() ; 
            switch(order_type){
               case OP_BUYSTOP:result = OrderDelete(order_number) ; 
               if(warn){
                  Alert("delete BUYSTOP sucessfully");
               }
            }
            if(result==false){
               if(warn){Alert("delete BUYSTOP failed");}
            }
         }
      }
   }
}

void close_buylimit(){
   int order_number = 0 ; 
   int order_type ; 
   int i ; 
   bool result = false ; 
   for(i=OrdersTotal()-1;i>=0;i--){
      if(OrderSelect(i,SELECT_BY_POS)){
         if(OrderSymbol()==currency&&OrderMagicNumber()==MAGIC){
            order_number = OrderTicket() ; 
            order_type = OrderType() ; 
            switch(order_type){
               case OP_BUYLIMIT:result = OrderDelete(order_number) ; 
               if(warn){
                  Alert("delete BUYLIMIT sucessfully");
               }
            }
            if(result==false){
               if(warn){Alert("delete BUYLIMIT failed");}
            }
         }
      }
   }
}

void close_sellstop(){
   int order_number = 0 ; 
   int order_type ; 
   int i ; 
   bool result = false ; 
   for(i=OrdersTotal()-1;i>=0;i--){
      if(OrderSelect(i,SELECT_BY_POS)){
         if(OrderSymbol()==currency&&OrderMagicNumber()==MAGIC){
            order_number = OrderTicket() ; 
            order_type = OrderType() ; 
            switch(order_type){
               case OP_SELLSTOP:result = OrderDelete(order_number) ; 
               if(warn){
                  Alert("delete SELLSTOP sucessfully");
               }
            }
            if(result==false){
               if(warn){Alert("delete SELLSTOP failed");}
            }
         }
      }
   }
}

void close_selllimit(){
   int order_number = 0 ; 
   int order_type ; 
   int i ; 
   bool result = false ; 
   for(i=OrdersTotal()-1;i>=0;i--){
      if(OrderSelect(i,SELECT_BY_POS)){
         if(OrderSymbol()==currency&&OrderMagicNumber()==MAGIC){
            order_number = OrderTicket() ; 
            order_type = OrderType() ; 
            switch(order_type){
               case OP_SELLLIMIT:result = OrderDelete(order_number) ; 
               if(warn){
                  Alert("delete SELLLIMIT sucessfully");
               }
            }
            if(result==false){
               if(warn){Alert("delete SELLLIMIT failed");}
            }
         }
      }
   }
}

void close_all(){
   int order_number = 0 ; 
   int order_type ; 
   int i ; 
   bool result = false ; 
   for(i=OrdersTotal()-1;i>=0;i--){
      if(OrderSelect(i,SELECT_BY_POS)){
         if(OrderSymbol()==currency&&OrderMagicNumber()==MAGIC){
            order_number = OrderTicket() ; 
            order_type = OrderType() ; 
            switch(order_type){
               case OP_BUYLIMIT:
               case OP_BUYSTOP:
               case OP_SELLLIMIT:
               case OP_SELLSTOP:result = OrderDelete(order_number) ; 
               if(warn){
                  Alert("delete all sucessfully");
               }
            }
            if(result==false){
               if(warn){Alert("delete all failed");}
            }
         }
      }
   }
}