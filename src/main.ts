var ort: typeof import("onnxruntime-web");

export { encodeImg as img, encodeText as text, initText, initImg, r };

var dev = true;
type AsyncType<T> = T extends Promise<infer U> ? U : never;
type SessionType = AsyncType<ReturnType<typeof import("onnxruntime-web").InferenceSession.create>>;
var text: SessionType;
var img: SessionType;
const imgW = 224,
    imgH = 224;

async function initText(x: {
    textPath: string;
    node?: boolean;
    dev?: boolean;
    ort?: typeof import("onnxruntime-web");
}) {
    if (x.ort) {
        ort = x.ort;
    } else {
        if (x.node) {
            ort = require("onnxruntime-node");
        } else {
            ort = require("onnxruntime-web");
        }
    }
    dev = x.dev;
    text = await ort.InferenceSession.create(x.textPath);
    return new Promise((rs) => rs(true));
}

async function initImg(x: { imgPath: string; node?: boolean; dev?: boolean; ort?: typeof import("onnxruntime-web") }) {
    if (x.ort) {
        ort = x.ort;
    } else {
        if (x.node) {
            ort = require("onnxruntime-node");
        } else {
            ort = require("onnxruntime-web");
        }
    }
    dev = x.dev;
    img = await ort.InferenceSession.create(x.imgPath);
    return new Promise((rs) => rs(true));
}

/** 主要操作 */
async function encodeImg(image: ImageData) {
    if (dev) console.time();
    let transposedData = beforeImg(image);
    const detResults = await runImg(transposedData, img);
    if (dev) {
        console.log(detResults);
        console.timeEnd();
    }
    return detResults.data;
}

/** 主要操作 */
async function encodeText(text: string) {
    if (dev) console.time();
    let tokens = beforeText(text);
    let nt: number[] = [];
    for (let i = 0; i < 52; i++) {
        nt.push(tokens[i] || 0);
    }
    const detResults = await runText(nt);
    if (dev) {
        console.log(detResults);
        console.timeEnd();
    }
    return detResults.data;
}

async function runImg(transposedData: number[][][], det: SessionType) {
    let x = transposedData.flat(Infinity) as number[];
    const detData = Float32Array.from(x);

    const detTensor = new ort.Tensor("float32", detData, [1, 3, imgH, imgW]);
    let detFeed = {};
    detFeed[det.inputNames[0]] = detTensor;

    const detResults = await det.run(detFeed);
    return detResults[det.outputNames[0]];
}

function data2canvas(data: ImageData, w?: number, h?: number) {
    let x = document.createElement("canvas");
    x.width = w || data.width;
    x.height = h || data.height;
    x.getContext("2d").putImageData(data, 0, 0);
    return x;
}

/**
 *
 * @param {ImageData} data 原图
 * @param {number} w 输出宽
 * @param {number} h 输出高
 */
function resizeImg(data: ImageData, w: number, h: number) {
    let x = data2canvas(data);
    let src = document.createElement("canvas");
    src.width = w;
    src.height = h;
    src.getContext("2d").scale(w / data.width, h / data.height);
    src.getContext("2d").drawImage(x, 0, 0);
    return src.getContext("2d").getImageData(0, 0, w, h);
}

function beforeImg(image: ImageData) {
    image = resizeImg(image, imgW, imgH);

    const transposedData = toImgInput(image, [0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]);
    if (dev) {
        let srcCanvas = data2canvas(image);
        document.body.append(srcCanvas);
    }
    return transposedData;
}

function toImgInput(image: ImageData, mean: number[], std: number[]) {
    const imagedata = image.data;
    const redArray: number[][] = [];
    const greenArray: number[][] = [];
    const blueArray: number[][] = [];
    let x = 0,
        y = 0;
    for (let i = 0; i < imagedata.length; i += 4) {
        if (!blueArray[y]) blueArray[y] = [];
        if (!greenArray[y]) greenArray[y] = [];
        if (!redArray[y]) redArray[y] = [];
        redArray[y][x] = (imagedata[i] / 255 - mean[0]) / std[0];
        greenArray[y][x] = (imagedata[i + 1] / 255 - mean[1]) / std[1];
        blueArray[y][x] = (imagedata[i + 2] / 255 - mean[2]) / std[2];
        x++;
        if (x == image.width) {
            x = 0;
            y++;
        }
    }

    return [blueArray, greenArray, redArray];
}

import { loadTokenizer, CLS_INDEX, SEP_INDEX } from "./bert_tokenizer";

function beforeText(text: string) {
    const tokenizer = loadTokenizer();
    const encoded = tokenizer.tokenize(text);
    console.log("encoded", encoded);
    return [CLS_INDEX, ...encoded, SEP_INDEX];
}

async function runText(data: number[]) {
    let l: bigint[] = [];
    for (let i of data) {
        l.push(BigInt(i));
    }
    const detData = BigInt64Array.from(l);

    const detTensor = new ort.Tensor("int64", detData, [1, data.length]);
    let detFeed = {};
    detFeed[text.inputNames[0]] = detTensor;

    const detResults = await text.run(detFeed);
    return detResults[text.outputNames[0]];
}

function r(img: number[], text: number[]) {
    let r = 0;
    if (img.length != 512 || text.length != 512) return r;
    img = toOne(img);
    text = toOne(text);
    for (let i = 0; i < 512; i++) {
        r += 100 * img[i] * text[i];
    }
    return r;
}

function toOne(l: number[]) {
    let n: number[] = [];
    let t = 0;
    for (let i of l) {
        t += i ** 2;
    }
    t = Math.sqrt(t);
    for (let i of l) {
        n.push(i / t);
    }
    return n;
}
