<template>
  <div>
    <h1>reactive 引用类型的响应式数据</h1>
    <h3>{{ person }}</h3>
    <button @click="changePerson">修改person</button>
    <button @click="person.k1.k2.k3 = 999">修改person的属性</button>
    <button @click="person.sex = '女'">添加属性</button>
    <button @click="delete person.name">删除属性</button>
    <hr />
    <h3>数组 ：{{ arr }}</h3>
    <button @click="arr.push(5)">push</button>
  </div>
</template>

<script setup>
//reactive ref都是用来定义vue3响应式数据
//ref 一般用于定义基本类型响应式数据
//reactive 定义引用类型的响应式数据
import {reactive} from 'vue'

const person = reactive({
  name: '小橘猫',
  age: 11,
  k1: {
    k2: {
      k3: 666,
    },
  },
})
console.log('person', person)
function changePerson() {
  person.name = '大花猫'
  person.age = 12
}

//创建原始数据
const cat = {
  name: '大花猫',
  age: 12,
}
//创建代理数据 代理对象本身 新增和删除属性都能被劫持到
const catProxy = new Proxy(cat, {
  //访问器 getters函数
  get(target, property, receiver) {
    //依赖收集
    //Reflect 隐射获取对象的数据
    return Reflect.get(target, property, receiver)
  },
  //设置器 setters函数
  set(target, property, newValue, receiver) {
    //派发更新
    return Reflect.set(target, property, newValue, receiver)
  },
})

console.log('catProxy', catProxy.age)
catProxy.age = 20
console.log('catProxy', catProxy.age)

//数据响应式数据
const arr = reactive([1, 2, 3, 4])
</script>

<style lang="scss" scoped></style>