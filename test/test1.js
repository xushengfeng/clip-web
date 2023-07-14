const x = require("../");

start();

async function start() {
    await x.initText({ textPath: "./m/vit-b-16.txt.fp32.onnx", dev: true, node: true });
    x.text("你好世界").then((data) => {
        console.log(data);
    });
}
