<template>
  <div>
    <h1>vue3工具函数</h1>
    <p>{{ person }}</p>
    <p>{{ name }}</p>
    <p>{{ age }}</p>
    <p>{{ sex }}</p>
    <button @click="person.name = '大花猫'">修改person</button>
    <button @click="name = '大花猫'">修改name</button>
    <h3>{{ readPerson }}</h3>
    <button @click="readPerson.name = '大花猫'">修改readonly</button>
  </div>
</template>

<script setup>
import {
  ref,
  isRef,
  unref,
  reactive,
  toRef,
  toRefs,
  isProxy,
  isReactive,
  isReadonly,
  readonly,
  toRaw,
  computed,
} from 'vue'

const num = ref(10)
//isRef判断是否是ref创建的数据
console.log('isRef(num)', isRef(num))
//unref 获取ref数据的value值
console.log('unref(num)', unref(num))

//toRef & toRefs 解构reactive响应式数据属性 拥有和源数据的响应式链接
const person = reactive({
  name: '小橘猫',
  age: 11,
  sex: true,
  address: {
    city: '成都市',
    point: [111, 222],
  },
})

const name = toRef(person, 'name')
const {age, sex} = toRefs(person)

//isProxy 判断数据是否为reactive or readonly创建的数据
const p1 = new Proxy(
  {},
  {
    get() {},
    set() {},
  }
)

console.log('isProxy(p1)', isProxy(p1)) //false
console.log('isProxy(person)', isProxy(person)) //true

//创建一个readonly数据
const readPerson = readonly({
  name: '小橘猫',
  age: 11,
})
console.log('isProxy(readPerson)', isProxy(readPerson)) //true
console.log('isReadonly(readPerson)', isReadonly(readPerson)) //true
console.log('isReactive(readPerson)', isReactive(person)) //true

//toRaw 获取reactive数据的原始值
console.log('toRaw(person)', toRaw(person))
</script>

<style lang="scss" scoped></style>