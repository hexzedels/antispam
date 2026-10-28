// Screenshots of the running service for the report and the deck.
// Usage: NODE_PATH=<dir with playwright> [CHROME_PATH=<chromium binary>] node scripts/screenshots.mjs [base_url]
import { createRequire } from "module";
const require = createRequire(import.meta.url);
const { chromium } = require("playwright");

const base = process.argv[2] || "http://localhost:7860";
const out = new URL("../reports/img/", import.meta.url).pathname;
const shots = [
  ["service_demo_spam.png", "btn-spam", "URGENT: Your account has been selected for a $5,000 cash reward! Click here http://claim-reward.example.biz to claim your prize now. Limited time offer, act now!"],
  ["service_demo_ham.png", "btn-spam", "Hi Anna,\n\nthanks for sending the slides. The evaluation section looks good, I only left two comments on the threshold part. Can we move our call to Thursday at 3 pm?\n\nBest,\nIvan"],
  ["service_demo_ru_spam.png", "btn-spam", "Поздравляем! Вы выиграли 500 000 рублей в нашей акции. Для получения выигрыша перейдите по ссылке http://priz-online.example.ru и оплатите комиссию 990 руб."],
  ["service_demo_category.png", "btn-cat", "Dear PayPal member, your account has been limited. Please verify your billing information at http://paypal-secure.example.com within 24 hours."],
];

const browser = await chromium.launch(process.env.CHROME_PATH ? { executablePath: process.env.CHROME_PATH } : {});
const page = await browser.newPage({ viewport: { width: 1100, height: 900 }, deviceScaleFactor: 2 });
for (const [file, button, text] of shots) {
  await page.goto(base + "/");
  await page.fill("#text", text);
  await page.click("#" + button);
  await page.waitForFunction(() => document.querySelector("#out").innerText.trim().length > 0);
  await page.waitForTimeout(300);
  await page.screenshot({ path: out + file, fullPage: true });
  console.log(file);
}
await page.goto(base + "/docs");
await page.waitForSelector(".opblock");
await page.screenshot({ path: out + "service_docs.png", fullPage: true });
console.log("service_docs.png");
await browser.close();
