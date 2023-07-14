const x = require("../");

start();

async function start() {
    await x.initImg({
        imgPath: "./m/vit-b-16.img.fp32.onnx",
        dev: true,
        node: true,
    });
    let img = document.createElement("img");
    img.src = "../x.webp";
    img.onload = () => {
        let canvas = document.createElement("canvas");
        canvas.width = img.width;
        canvas.height = img.height;
        canvas.getContext("2d").drawImage(img, 0, 0);
        x.img(canvas.getContext("2d").getImageData(0, 0, img.width, img.height)).then((data) => {
            
        });
    };
}
