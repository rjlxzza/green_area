function echarts_map() {
    // 初始化 ECharts 地图实例
    var myChart = echarts.init(document.getElementById('map_1'));

    // 地图数据文件路径
    var anhui = "js/44.json";

    // 使用 jQuery 获取地图数据
    $.get(anhui, function(geoJson) {
        // 注册地图数据
        echarts.registerMap('anhui', geoJson);

        // 设置图表的配置项和数据显示图表
        myChart.setOption({
            series: [{
                type: 'map', // 指定图表类型为地图
                map: 'anhui' // 使用注册的地图
            }]
        });

        // 安徽省城市经纬度
        var geoCoordMap = {
            '合肥市': [117.283042, 31.86119],
            '芜湖市': [118.376451, 31.326319],
            '蚌埠市': [117.363228, 33.139667],
            '淮南市': [117.018329, 32.647574],
            '马鞍山市': [118.507906, 31.689362],
            '淮北市': [116.794664, 33.971707],
            '铜陵市': [118.316264, 32.303627],
            '安庆市': [116.333551, 30.70883],
            '黄山市': [118.317325, 29.709239],
            '滁州市': [118.316264, 32.303627],
            '阜阳市': [115.819729, 32.896969],
            '宿州市': [116.984084, 33.633891],
            '六安市': [116.507676, 30.952889],
            '亳州市': [115.782939, 33.869338],
            '池州市': [117.489157, 30.656037],
            '宣城市': [118.757995, 30.545667]
        };

        // 示例城市数据
        var goData = [
            { name: '合肥市', value: 8 },
            { name: '蚌埠市', value: 5 },
            // 其他城市数据...
            { name: '安庆市', value: 5 },
            { name: '宣城市', value: 5 }
        ];

        // 计算总人数
        var goTotal = 0; //计算总人数
        goData.forEach(function(item, i) {
            goTotal += item.value;
        });

        // 转换数据格式
        var convertData = function(name, data) {
            var res = [];
            for (var i = 0; i < data.length; i++) {
                var fromCoord = geoCoordMap[name];
                var toCoord = geoCoordMap[data[i].name];
                if (fromCoord && toCoord) {
                    res.push({
                        coords: [fromCoord, toCoord]
                    });
                }
            }
            return res;
        };
        var series = [];
        [
            ['合肥市', goData],
            // 其他城市数据...
        ].forEach(function(item, i) {
            series.push({
                name: item[0],
                type: 'lines',
                zlevel: 2,
                effect: {
                    show: true,
                    period: 6,
                    trailLength: 0.1,
                    symbol: 'arrow', //标记类型
                    symbolSize: 10
                },
                lineStyle: {
                    normal: {
                        width: 1,
                        opacity: 0.4,
                        curveness: 0.2, //弧线角度
                        color: 'rgba(255,255,255,.1)'
                    }
                },
                data: convertData(item[0], item[1])
            }, { //终点
                name: item[0],
                type: 'scatter',
                coordinateSystem: 'geo',
                zlevel: 2,
                label: {
                    normal: {
                        show: true,
                        color: 'rgba(255,255,255,.5)',
                        position: 'right',
                        formatter: '{b}'
                    }
                },
                symbol: 'circle',
                symbolSize: function(val) {
                    return val[2] * 50 / goTotal;
                },
                itemStyle: {
                    normal: {
                        show: true
                    }
                },
                data: item[1].map(function(dataItem) {
                    console.log(dataItem);
                    return {
                        name: dataItem.name,
                        value: geoCoordMap[dataItem.name].concat([dataItem.value])
                    };
                })
            });
        });

        // 地图配置
        var option = {
            title: {
                text: '安徽省核心业务覆盖',
                left: 'center',
                top: '0',
                textStyle: {
                    color: '#fff',
                    fontSize: 24,
                }
            },
            tooltip: {
                trigger: 'item',
                formatter: "{b}"
            },
            visualMap: {
                show: false,
                min: 0,
                max: 100,
                color: ['#fff']
            },
            geo: {
                map: 'anhui',
                zoom: 1,
                label: {
                    normal: {
                        show: true,
                        textStyle: {
                            color: 'rgba(255,255,255,.3)'
                        }
                    },
                    emphasis: {
                        textStyle: {
                            color: '#fff'
                        }
                    }
                },
                roam: false,
                itemStyle: {
                    normal: {
                        areaColor: '#4256ff',
                        borderColor: '#404a59'
                    },
                    emphasis: {
                        areaColor: '#2539f5'
                    }
                }
            },
            series: series
        };

        // 设置地图配置
        myChart.setOption(option);

        // 添加点击事件处理逻辑
        // 给 myChart 对象绑定点击事件监听器
        myChart.on('click', function(params) {
            // 在控制台打印点击事件的参数，用于调试
            console.log(params);


                // 获取被点击的地区名称
                var cityName = params.name;

                // 判断是否点击了“芜湖市”
                if (params.name === "芜湖市") {
                    // 如果点击了“芜湖市”，跳转到芜湖市地图
                    window.location.href = '../005 可视化监控管理/index.html';
                }
        });

        // 监听窗口大小变化，自适应地图大小
        window.addEventListener("resize", function() {
            myChart.resize();
        });
    });
}



// 当网页加载完毕后，初始化地图
$(window).load(function() {
    echarts_map();
});