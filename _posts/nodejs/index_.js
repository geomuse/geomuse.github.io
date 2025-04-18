const express = require("express");

const app = express();

app.get("/",function(req,res){
    res.send("I\'m geo") ; 
})

app.get("/get",function(req,res){
    res.send("here you go.")
})

app.listen(3000,function(){
    console.log('server started.');
});