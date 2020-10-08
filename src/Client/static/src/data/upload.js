import fetch from './services/fetch'

import apiUrl from './api-route'

const register = async function (imageSrc, name, id) {
    name = String(name);
    id = String(id);
    name = name.replace("_", "");
    id = id.replace("_", "");
    const body = name + "_" + id + "_" + imageSrc;
    console.log({body: body})
    const response = await fetch.post_binary(apiUrl.upload, body)
    return response
};

export default {register}