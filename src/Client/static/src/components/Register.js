import React, { Component, useEffect, useCallback, memo } from "react";
import { makeStyles, useTheme } from "@material-ui/core/styles";
import Paper from "@material-ui/core/Paper";
import Button from "@material-ui/core/Button";
import Divider from "@material-ui/core/Divider";
import { withRouter } from "react-router";;
import Grid from "@material-ui/core/Grid";
import Webcam from "react-webcam";
import FormControl from "@material-ui/core/FormControl";
import InputLabel from "@material-ui/core/InputLabel";
import Select from "@material-ui/core/Select";
import MenuItem from "@material-ui/core/MenuItem";
import Dialog from '@material-ui/core/Dialog';
import DialogActions from '@material-ui/core/DialogActions';
import DialogContent from '@material-ui/core/DialogContent';
import DialogContentText from '@material-ui/core/DialogContentText';
import DialogTitle from '@material-ui/core/DialogTitle';
import Switch from '@material-ui/core/Switch';
import { green } from "@material-ui/core/colors";
import CircularProgress from "@material-ui/core/CircularProgress";

// Local import
import upload from "./../data/upload";
import staff from "../data/staff";

const useStyles = makeStyles(theme => ({
    textField: {
        margin: "10px",
        padding: "10px",
        marginTop: theme.spacing(0),
        marginBottom: theme.spacing(0)
    },
    button: {
        margin: theme.spacing(2)
    },
    container: {
        display: "flex",
        flexWrap: "wrap"
    },
    paper: {
        maxWidth: "100%",
        margin: `${theme.spacing(1)}px auto`,
        padding: theme.spacing(1),
        align: "center"
    },
    title: {
        fontWeight: "bold",
        fontSize: "1.5rem",
        marginVertical: "1em",
        textAlign: "center"
    },
    dense: {
        marginTop: theme.spacing(2)
    },
    formControl: {
        margin: theme.spacing(1),
        minWidth: 120,
        maxWidth: 300,
    },
    buttonProgress: {
        color: green[500],
        position: 'absolute',
        top: '50%',
        left: '50%',
        marginTop: -12,
        marginLeft: -12,
    },
    formControl: {
        marginTop: theme.spacing(2),
        minWidth: 120,
    },
    formControlLabel: {
        marginTop: theme.spacing(1),
    },
}));

function Register(props) {

    const classes = useStyles();

    const [image, setImage] = React.useState("");
    const [state, setState] = React.useState({
        name: "",
        id: "",
        department: ""
    });
    const [staffList, setStaffList] = React.useState([]);
    const [showInputForm, setShowInputForm] = React.useState(true);
    const [userID, setUserID] = React.useState("");


    const handleChangeDropdown = event => {
        var userid = event.target.value;
        var userinfo = getUserInfoFromID(userid);
        setUserID(String(event.target.value));
        setState(userinfo);
    };


    const webcamRef = React.useRef(null);

    // load  for init
    useEffect(
        () => {
            console.log("Start get staff list");
            staff.getAll().then(res => {
                console.log({ staff: res });
                setStaffList(Object.values(res));
            })
            clearTimeout(timer.current);
        }, []);

    const videoConstraints = props.settings;

    const [loading, setLoading] = React.useState(false);
    const [success, setSuccess] = React.useState(false);
    const [enableWebcam, setEnableWebcam] = React.useState(true);
    const [showImg, setShowImg] = React.useState(false);
    const [registerVertification, setRegisterVertification] = React.useState(false);

    const timer = React.useRef();


    const registerPicture = () => {
        setEnableWebcam(false);
        setShowImg(true);
        setRegisterVertification(false);
        const name = state.name;
        const id = state.id;
        console.log(state.name + "_" + state.id);
        if (!loading) {
            setSuccess(false);
            setLoading(true);
        }
        upload.register(image, name, id).then(res => {
            return res.json();
        }).then(function (data) {
            setSuccess(true);
            setLoading(false);
            setShowImg(false);
            if (data.errcode == 0) {
                alert("登録が成功しました。")
                setState({
                    name: "",
                    id: "",
                    department: ""
                })
                setUserID("");
            } else {
                alert("登録が失敗しました。" + data.msg);
            }
            setEnableWebcam(false);
            setShowInputForm(false);
            // re render parent
            props.rerenderParentCallback();
        })

    }


    const handleSendPictureButton = () => {
        if (enableWebcam) {
            if(state.id===""){
                alert("IDを選択してください。");
                return;
            }
            const imageSrc = webcamRef.current.getScreenshot();
            setImage(imageSrc);
            setShowImg(true);
            setShowInputForm(false);
            setRegisterVertification(true);
        }
        else {
            alert("カメラをオンにしてください。");
        }
    }

    const handleCancelButton = () => {
        setEnableWebcam(false);
        // re render parent
        props.rerenderParentCallback();
    }


    const getUserInfoFromID = (userid) => {
        userid = String(userid);
        for (var index in staffList) {
            var item = staffList[index];
            if (item.id == userid) {
                return item;
            }
        }
        return {
            name: "IDが選択されていません。",
            department: "IDが選択されていません。",
            id: "IDが選択されていません。"
        }
    }
    const renderUserInfo = (userid) => {
        if (userid == "") {
            userid = "IDが選択されていません。"
        }
        var userInfo = getUserInfoFromID(userid);
        return (
            <div className={classes.textField}>
                ID: {userid} 氏名: {userInfo.name} 部署：{userInfo.department}
            </div>
        )
    }

    const renderPersonInfoDropdownForm = () => {
        return (
            <div>
                <FormControl className={classes.formControl}>
                    <InputLabel id="controlled-open-select-label">ID</InputLabel>
                    <Select
                        labelId="controlled-open-select-label"
                        id="controlled-open-select"
                        value={userID}
                        onChange={handleChangeDropdown}
                    >
                        {staffList.map(staff => (
                            <MenuItem key={staff.id} value={staff.id}>
                                {staff.id}
                            </MenuItem>
                        ))}
                    </Select>
                </FormControl>
                <Button color="primary"
                    variant="contained"
                    onClick={handleSendPictureButton}
                    className={classes.button}
                    disabled={loading}>
                    送信</Button>
                <Button onClick={handleCancelButton}
                    color="primary"
                    variant="contained"
                    className={classes.button}
                    disabled={loading}
                >キャンセル</Button>
            </div>
        )
    }

    const closeRegisterVer = () => {
        setRegisterVertification(false);
        setShowInputForm(true);
    }

    const renderRegisterVertificationDialog = () => {
        return (
            <Dialog
                open={registerVertification}
                onClose={closeRegisterVer}
                aria-labelledby="alert-dialog-title"
                aria-describedby="alert-dialog-description"
            >
                <DialogTitle id="alert-dialog-title">{"以下の画像を送信しますか?"}</DialogTitle>
                <DialogContent>
                    <DialogContentText id="alert-dialog-description">
                        {showImg ?
                            <img width="320" height="240" src={`${image}`} />
                            : <div>画像が存在しません。</div>}
                    </DialogContentText>
                </DialogContent>
                <DialogActions>
                    <Button onClick={closeRegisterVer} color="primary">
                        いいえ、撮り直します。
          </Button>
                    <Button onClick={registerPicture} color="primary" autoFocus>
                        はい、送信します。
          </Button>
                </DialogActions>
            </Dialog>
        )
    }

    return (
        <Paper className={classes.paper}>
            <div className={classes.title}>
                ユーザー画像情報登録
            </div>
            <Grid
                container
                direction="column"
                justify="flex-start"
                alignItems="center"
                spacing={3}
            >
                <Grid item xs={12}>
                    {enableWebcam ? <Webcam
                        audio={false}
                        height={videoConstraints.height}
                        ref={webcamRef}
                        screenshotFormat="image/jpeg"
                        width={videoConstraints.width}
                        videoConstraints={videoConstraints}
                    /> : ""}

                </Grid>
                {loading && <CircularProgress size={72} className={classes.buttonProgress} />}
                <Divider />
                {showInputForm ? renderUserInfo(userID) : ""}
                {showInputForm ? renderPersonInfoDropdownForm() : ""}
                {renderRegisterVertificationDialog()}
                <Divider />
            </Grid>
        </Paper>

    );
}

export default withRouter(Register);


