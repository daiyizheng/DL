<template>
  <div>
    <h1>toRef & toRefs</h1>
    <p>{{ person }}</p>
    <p>name:{{ name }}</p>
    <p>age:{{ age }}</p>
    <p>sex:{{ sex }}</p>
    <button @click="changePerson">修改person</button>
    <button @click="name = '小黑猫'">修改name</button>
  </div>
</template>

<script setup>
import { reactive, toRef, toRefs } from "vue";

//解构reactive的响应式数据 转换为 ref响应式
const person = reactive({
  name: "小橘猫",
  age: 11,
  sex: "女",
});
//toRef 获取的响应式数据 和源数据 具备响应式链条
const name = toRef(person, "name");

//toRefs 获取的响应式数据 和源数据 具备响应式链条
const { age, sex } = toRefs(person);

function changePerson() {
  person.name = "大花猫";
  person.age = 12;
  person.sex = "男";
}
</script>

<style lang="scss" scoped></style>
