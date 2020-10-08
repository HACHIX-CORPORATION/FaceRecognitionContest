this server run with https, working on macos safari  
**how to create key**  
`openssl genrsa -aes128 2048 > server_secret.key`  
`openssl req -new -key server_secret.key > server_pub.csr`  
`openssl x509 -in server_pub.csr -days 365 -req -signkey server_secret.key > cert.crt`


**memo**  
pem pw: hachix

# ssl
openssl rsa -in **.key **.key
