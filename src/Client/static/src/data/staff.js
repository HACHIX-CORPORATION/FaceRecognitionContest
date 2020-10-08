import fetch from './services/fetch'

import apiUrl from './api-route'

const register = async function (body) {
    const response = await fetch.post_binary(apiUrl.staff, body);
    return response
};

const getAll = async function(){
    const response = await fetch.get(apiUrl.staff + '?item=all');
    return response;
}

export default {register, getAll}