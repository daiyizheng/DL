<template>
  <div>
    <h1>基本语法</h1>
    <p>我是p标签</p>
    <h3>胡须语法</h3>
    <button @click="$router.push({ name: 'user', params: { name: '小橘猫' } })">
      哈哈
    </button>
    <h3>变量:{{ num1 }}</h3>
    <h3>函数:{{ getMsg() }}</h3>
    <h3>四则运算:{{ num1 + num2 }}</h3>
    <h3>三目运算:{{ false ? 40 : 60 }}</h3>
    <h3>短路运算符:{{ true && "你好啊" }}</h3>
    <h3>正则表达式:{{ /www/.test("wwwwww") }}</h3>
  </div>
</template>

<script setup>
import { ref } from "vue";

//定义在全局的数据和方法 自动被模板解包 可以直接使用

//胡须语法 {{表达式}}
const num1 = 20;

//函数
function getMsg() {
  return "你好啊";
}

//四则运算
const num2 = 30;
</script>

<style lang="scss" scoped></style>
