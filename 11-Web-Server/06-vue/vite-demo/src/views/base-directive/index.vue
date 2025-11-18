<template>
  <div>
    <h1>基本指令</h1>
    <h3>v-text & v-html</h3>
    <div v-text="msg"></div>
    <div v-html="msg"></div>
    <hr />
    <h3>v-show</h3>
    <p v-show="flag">我是一个p标签</p>
    <hr />
    <h3>v-if</h3>
    <p v-if="flag">我是一个p标签</p>
    <h3>v-if v-else-if v-else</h3>
    <p v-if="score >= 90">优秀</p>
    <p v-else-if="score >= 70">良好</p>
    <p v-else-if="score >= 60">及格</p>
    <p v-else>不及格</p>
    <button v-if="false">编辑</button>
    <button v-else>完成</button>
    <hr />
    <h3>v-for循环指令</h3>
    <!-- item 形参 循环的元素 -->
    <!-- index 形参 循环的索引 -->
    <!-- :key="基本类型的唯一值" -->
    <div :key="item.name" v-for="(item, index) in shopData">
      {{ index }} -- {{ item.name }}
    </div>
    <hr />
    <h3>v-model 双向数据绑定指令</h3>
    <div class="form">
      <p class="form-item">
        <input type="text" v-model="name" />
        {{ name }}
      </p>
      <p class="form-item">
        <input type="radio" v-model="sex" value="男" />
        男
        <input type="radio" v-model="sex" value="女" />
        女---{{ sex }}
      </p>
      <p>
        <input type="checkbox" v-model="hobs" value="吃饭" />
        吃饭
        <input type="checkbox" v-model="hobs" value="睡觉" />
        睡觉
        <input type="checkbox" v-model="hobs" value="听音乐" />
        听音乐
        <input type="checkbox" v-model="hobs" value="打游戏" />
        打游戏 -- {{ hobs }}
      </p>
      <p class="form-item">
        <select v-model="city">
          <option value="成都市">成都市</option>
          <option value="武汉市">武汉市</option>
          <option value="广州市">广州市</option>
        </select>
        --{{ city }}
      </p>
      <p class="form-item">
        <button @click="show">展示数据</button>
      </p>
    </div>
  </div>
</template>

<script setup>
import { ref } from "vue";
//指令 v-xx='表达式'作用 帮助操作dom
//v-text 渲染内容 相当于innerText
//v-html 解析模板 相当于innerHtml
const msg = "<p>我是p标签</p>";

//v-show 切换display 值判断为false display:none
const flag = false;
//v-if 判断节点是否渲染

//考试成绩 >90优秀 >70良好 >=60及格 不及格
const score = 50;

//v-for 循环指令
const shopData = [
  { name: "小衣服", price: 300 },
  { name: "小裤子", price: 200 },
  { name: "小鞋子", price: 400 },
  { name: "小帽子", price: 500 },
];

//ref 创建基本类型的响应式数据
const name = ref("小橘猫");

function show() {
  console.log("name.value", name.value);
}

//性别
const sex = ref("女");

//爱好
const hobs = ref(["吃饭", "睡觉"]);

//城市
const city = ref("成都市");
</script>

<style lang="scss" scoped></style>
