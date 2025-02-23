package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"

	"github.com/gorilla/mux"
)

func homeHandler(w http.ResponseWriter, r *http.Request) {
	fmt.Fprintf(w, "欢迎访问 Go Web 服务器！")
}

func apiHandler(w http.ResponseWriter, r *http.Request) {
	response := map[string]string{"message": "Hello, API!"}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func main() {
	r := mux.NewRouter()
	r.HandleFunc("/", homeHandler).Methods("GET")
	r.HandleFunc("/api", apiHandler).Methods("GET")

	fmt.Println("服务器启动，监听端口 8080...")
	log.Fatal(http.ListenAndServe(":8080", r))
}
