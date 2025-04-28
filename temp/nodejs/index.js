x = require('./lib.js')

const fs = require('fs')

fs.writeFile('./test','hello writing file',function(err){
    if(err){
        console.log('error');
    }else{
        console.log('successful');
    }
});

fs.readFile('./test',{encoding:'utf-8'},function(error,data){
    if(error){
        console.log('error')
    }else{
        console.log(data)
    }
})

console.log("Hello I\'m geo.")
x(1,2)