const express = require('express');
const app = express();

app.use(express.static("index"));

app.get('/', (req, res) => {
    res.send('Hello World');
});

app.get('/user', (req, res) => {
    res.send('User Page');
});

app.listen(3000, () => {
    console.log('Server is running on port 3000');
});