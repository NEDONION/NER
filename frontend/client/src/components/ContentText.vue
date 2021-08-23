<template>
  <div class="py-6">
    <h1 class="text-4xl tracking-wide text-gray-800 border-b mb-4">
      实体抽取
    </h1>

    <div class="flex flex-row">
      <div class="text-gray-700" style="width:100%;">
        <div>
          <textarea
            v-model="content"
            cols="20"
            rows="8"
            style="width:100%; padding:15px; border:1px solid #f1f1f1;"
          ></textarea>
        </div>
      </div>
    </div>
    <div style="margin-top:30px; margin-bottom:30px;">
      <button
        class="bg-green-500 hover:bg-green-700 text-white font-bold py-2 px-4 rounded"
        @click.prevent="entityExtract"
      >
        实体抽取
      </button>
    </div>

    <div style="marign-bottom:30px; width:100%;">
      <div style="display:inline-block; width:100%;">
        <p v-html="results"></p>
      </div>
    </div>

    <div class="vld-parent">
      <loading
        :active.sync="isLoading"
        :can-cancel="false"
        :is-full-page="fullPage"
      ></loading>
    </div>
  </div>
</template>

<script>
import axios from "axios";
import Loading from "vue-loading-overlay";
// Import stylesheet
import "vue-loading-overlay/dist/vue-loading.css";

export default {
  name: "BooksTable",
  props: {
    heading: String,
    msg: String
  },
  data() {
    return {
      flag: true,
      books: [],
      showModal: false,
      modalState: {},
      rowEdited: -1,
      content:
        "6月17日， 神舟十二号在酒泉卫星发射中心准时点火发射，顺利将聂海胜、刘伯明、汤洪波3名航天员送入太空。",
      results: "",
      reportUrl: "",
      isLoading: false,
      fullPage: true
    };
  },
  components: {
    Loading
  },
  methods: {
    entityExtract() {
      this.isLoading = true;
      // this.bus.$emit('loading', true);
      axios
        //.get(`${process.env.VUE_APP_BACKEND_API}/`)
        .get(
          "http://127.0.0.1:5050/extract?content=" + this.content
          // { content: this.content },
          // {
          //   headers: {
          //     "Content-type": "application/json"
          //   },
          //   changeOrigin: true
          // }
        )
        .then(response => {
          console.log(response.data);
          let jsontext = response.data["result"];
          let a = response.data["content"];
          if (jsontext["PER"].length > 0) {
            for (let i = 0; i < jsontext["PER"].length; i++) {
              //console.log(i)
              a = a.replace(
                jsontext["PER"][i],
                '<span class="PER">' + jsontext["PER"][i] + "PER</span>"
              );
              // $("#container").append(re_a);
              //console.log(re_a)
            }
          }
          if (jsontext["LOC"].length > 0) {
            for (let i = 0; i < jsontext["LOC"].length; i++) {
              // console.log(i)
              a = a.replace(
                jsontext["LOC"][i],
                '<span class="LOC">' + jsontext["LOC"][i] + "LOC</span>"
              );
              // $("#container").append(re_a);
              //console.log(re_a)
            }
          }
          if (jsontext["ORG"].length > 0) {
            for (let i = 0; i < jsontext["ORG"].length; i++) {
              a = a.replace(
                jsontext["ORG"][i],
                '<span class="ORG">' + jsontext["ORG"][i] + "ORG</span>"
              );
            }
          }
          if (jsontext["EQU"].length > 0) {
            for (let i = 0; i < jsontext["EQU"].length; i++) {
              // console.log(i)
              a = a.replace(
                jsontext["EQU"][i],
                '<span class="EQU">' + jsontext["EQU"][i] + "EQU</span>"
              );
            }
          }
          console.log(a);
          this.results = a;
          this.isLoading = false;
        })
        .catch(error => {
          if (error.response) {
            console.log(error.response.data);
            console.log(error.response.status);
            console.log(error.response.headers);
          } else {
            console.log("Error", error.message);
          }
          console.log(error.config);
        });
    }
  },
  mounted() {
    //this.onButtonPress();
    // this.joinExtract();
    // this.onFusion();
  }
};
</script>

<style>
.PER {
  background: #f1a899;
  padding: 5px;
  border-radius: 5px;
}
.LOC {
  background: #449d44;
  padding: 5px;
  border-radius: 5px;
}
.ORG {
  background: #f0ad4e;
  padding: 5px;
  border-radius: 5px;
}
.EQU {
  background: #ce8483;
  padding: 5px;
  border-radius: 5px;
}
</style>
