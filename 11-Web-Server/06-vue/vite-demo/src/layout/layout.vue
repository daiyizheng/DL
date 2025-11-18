<template>
  <div class="layout df">
    <div class="navs fn df fdc">
      <router-link
        v-for="item in routes.filter((item) => item.path !== '/')"
        :to="item.path"
        :key="item.path"
        class="link df jcc aic mb10"
      >
        {{ item.path }}
      </router-link>
    </div>

    <div class="app-main f1 p15">
      框架
      <router-view></router-view>
    </div>
  </div>
</template>

<script setup>
import { routes } from "@/route/index";
console.log(routes);
// 逻辑在这里
</script>

<style scoped lang="scss">
.layout {
  height: 100%;
  .navs {
    width: 140px;
    background: antiquewhite;
    .link {
      height: 40px;
      background: darkcyan;
      color: white;
    }
  }
}

.app-main {
  overflow-y: scroll;
}
/* 样式在这里 */
</style>
