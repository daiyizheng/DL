<template>
  <div class="page-css">
    <h1>css作用域</h1>
    <div class="box"></div>
  </div>
</template>

<script setup>
import {ref, watch} from 'vue'

const num = ref([])

watch(
  num,
  async () => {
    await getNum()
  },
  {
    deep: true,
  }
)

let i = 0
setInterval(() => {
  console.log('++i', ++i)
}, 1000)

setTimeout(() => {
  num.value.push(10)
}, 1000)
setTimeout(() => {
  num.value.push(20)
}, 1500)
setTimeout(() => {
  num.value.push(30)
}, 2000)

function getNum() {
  return new Promise(reslove => {
    setTimeout(() => {
      console.log('num.value', num.value)
      reslove()
    }, 1000)
  })
}
</script>

<style lang="scss" scoped>
.box {
  width: 100px;
  height: 100px;
  border: 4px solid #666;
}

.page-css {
  color: blue;
}
</style>