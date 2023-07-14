const x = require("../");

start();

async function start() {
    await x.initImg({
        imgPath: "./m/vit-b-16.img.fp32.onnx",
        dev: true,
        node: true,
    });
    await x.initText({ textPath: "./m/vit-b-16.txt.fp32.onnx", dev: true, node: true });
    let img = document.createElement("img");
    img.src = "../a9.png";
    img.onload = () => {
        let canvas = document.createElement("canvas");
        canvas.width = img.width;
        canvas.height = img.height;
        canvas.getContext("2d").drawImage(img, 0, 0);
        let l = [];
        x.img(canvas.getContext("2d").getImageData(0, 0, img.width, img.height)).then((data) => {
            for (let i of ["杰尼龟", "妙蛙种子", "小火龙", "皮卡丘"]) {
                x.text(i).then((tdata) => {
                    let r = x.r(data, tdata);
                    l.push(r);
                    let t = 0;
                    for (let i of l) {
                        t += Math.E ** i;
                    }
                    let nl = [];
                    for (let i of l) {
                        nl.push(Math.E ** i / t);
                    }
                    console.log(i, nl);
                });
            }
        });
    };
}
