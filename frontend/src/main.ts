import { createApp } from 'vue'
import 'katex/dist/katex.min.css'
import './styles/main.css'
import App from './App.vue'
import { pinia } from './stores/pinia'
import { router } from './router'

createApp(App).use(pinia).use(router).mount('#app')
