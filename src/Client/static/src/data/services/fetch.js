import fetch from 'isomorphic-unfetch'

const get = async function (url) {
    const response = await fetch(url, {
        method: 'GET',
        headers: {'Content-Type': 'application/json'},
    })


    if (response.status !== 200) {
        let error = new Error("get ko thanh cong")
        throw error
    }
    ;

    const res = await response.json();
    return res
};

const post = async function (url, body) {
    const bod = JSON.stringify(body)
    const response = await fetch(url , {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(body)
    })

    if (response.status !== 200) {
        let error = new Error("post ko thanh cong")
        throw error
    }
    ;

// const res= await response.json();
    return response
};

const post_binary = async function (url, body) {
    const response = await fetch(url , {
        method: 'POST',
        headers: {'Content-Type': 'application/binary'},
        body: body
    })

    if (response.status !== 200) {
        let error = new Error("post binary ko thanh cong")
        throw error
    }
    ;

// const res= await response.json();
    return response
};

const remove = async function (url) {
    console.log("in fetch remove")
    const response = await fetch(url, {
        method: 'DELETE',
        headers: {'Content-Type': 'application/json'},
    })

    if (response.status !== 200) {
        let error = new Error("delete ko thanh cong")
        throw error
    }
    ;

// const res= await response.json();
    return response
};

const update = async function (url, body) {
    const bod = JSON.stringify(body)
    console.log("in fetch update")
    const response = await fetch(url, {
        method: 'PUT',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(body)
    })

    if (response.status !== 200) {
        let error = new Error("Update ko thanh cong")
        throw error
    }
    ;

    // const res= await response.json();
    return response
};


export default {get, post, post_binary, remove, update}