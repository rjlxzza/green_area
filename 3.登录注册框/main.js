/**
 * id 消息id
 * userId 用户id
 * content 内容
 * createTime 创建时间
 */
const express = require('express')
const bodyParser = require('body-parser')
const mysql = require('mysql2')
const app = express()

// 配置解析post请求参数
app.use(bodyParser.urlencoded({ extended: false }))
app.use(bodyParser.json())

// 连接数据库
let db = mysql.createConnection({
    host: 'lacalhost',
    port: '3306',
    user: 'root',
    password: 123456,
    database: 'test'
})

let data = []

/*
CREATE TABLE IF NOT EXISTS `message`(
    `id` INT UNSIGNED AUTO_INCREMENT,
    `userId` VARCHAR(100) NOT NULL,
    `content` VARCHAR(40) NOT NULL,
    `createtime` DATE,
    PRIMARY KEY ( `id` )
)ENGINE=InnoDB DEFAULT CHARSET=utf8;
*/

// 添加消息
app.post('/', (req, res) => {
    console.log(res.body);
    let sql = 'insert into message set userId=?,content=?,createTime=?'
    db.query(sql, [req.body.userId, req.body.content, req.body.createTime], (err, result) => {
        if (result.affectedRows == 1) {
            res.send({
                message: '数据插入成功',
                code: 200,
            })
        }
    })
})

// 获取聊天记录
app.get('/', (req, res) => {
    // 执行 SQL 查询，从 message 表中选取所有数据
    db.query('select * from message', (err, result) => {
        // 打印查询结果到控制台
        console.log(result);
        // 向客户端发送响应
        res.send({
            message: '获取成功',  // 提示信息
            code: 200,           // 状态码
            data: result         // 查询结果数据
        })
    })
})

// 启动服务器，监听 8080 端口
app.listen(8080, () => {
    // 服务器启动成功后，在控制台打印提示信息
    console.log('服务器创建成功');
})