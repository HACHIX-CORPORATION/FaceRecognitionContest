import fetch from './services/fetch'

import apiUrl from './api-route'

const register = async function (body) {
    const response = await fetch.post_binary(apiUrl.engine + "/predict", body);
    return response
};

const postFeedback= async function (body){
    const response = await fetch.post(apiUrl.engine + "/feedback", body);
    return response;
}

const getFeedback = async function(){
    const res = await fetch.get(apiUrl.engine+"/feedback");
    return res;
}

export default {register, postFeedback, getFeedback }