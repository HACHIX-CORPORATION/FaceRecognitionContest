import fetch from 'isomorphic-unfetch'
import apiUrl from '../api-route'
import store from 'store'

const getToken=async function(){
    const response = await fetch(apiUrl.token+"/", {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
    })

    if (response.status!==200){
        let error= new Error("get ko thanh cong")
        throw error
    };

    const res= await response.json();

    window.jwtToken = res.token;
    store.set('token', res.token)
    console.log('auth thanh cong')
    return res
};


const AuthGet=async function(url){
    const response = await fetch(url, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
    })

    if (response.status!==200){
        let error= new Error("get ko thanh cong")
        throw error
    };

    const res= await response.json();

    return res
};




export default {getToken,AuthGet}