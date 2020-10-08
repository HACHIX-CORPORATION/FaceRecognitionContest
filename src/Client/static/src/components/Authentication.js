import React, { useEffect, useRef, useCallback } from "react";
import { makeStyles, useTheme } from "@material-ui/core/styles";
import Paper from "@material-ui/core/Paper";
import Button from "@material-ui/core/Button";
import Divider from "@material-ui/core/Divider";
import { withRouter } from "react-router";
import Grid from "@material-ui/core/Grid";
import FormControl from "@material-ui/core/FormControl";
import InputLabel from "@material-ui/core/InputLabel";
import Select from "@material-ui/core/Select";
import MenuItem from "@material-ui/core/MenuItem";
import Dialog from '@material-ui/core/Dialog';
import DialogActions from '@material-ui/core/DialogActions';
import DialogContent from '@material-ui/core/DialogContent';
import DialogContentText from '@material-ui/core/DialogContentText';
import DialogTitle from '@material-ui/core/DialogTitle';
import { green } from "@material-ui/core/colors";
import CircularProgress from "@material-ui/core/CircularProgress";
import Webcam from "react-webcam";

// Local import
import authentication from "../data/authentication";
import staff from "../data/staff";



const useStyles = makeStyles(theme => ({
    root: {
        flexGrow: 1,
        overflow: "hidden",
        padding: theme.spacing(0, 3)
    },
    button: {
        margin: theme.spacing(2)
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

function Authentication(props) {

    const classes = useStyles();

    const [image, setImage] = React.useState("");
    const [authenticationInfo, setAuthInfo] = React.useState("");

    const [staffListArray, setStaffListArray] = React.useState([]);

    const [showAuthenticationInfo, setShowAuthInfo] = React.useState(false);


    const videoConstraints = props.settings;

    // load  for init
    useEffect(
        () => {
            console.log("Start get staff list");
            staff.getAll().then(res => {
                console.log({ staff: res });
                setStaffListArray(Object.values(res));
            })
            clearTimeout(timer.current);
        }, []);

    const [loading, setLoading] = React.useState(false);
    const [successAuth, setSuccessAuth] = React.useState(false);
    const [open, setOpen] = React.useState(false);
    const [openSelectDialog, setOpenSelectDialog] = React.useState(false);
    const [dialogMsg, setDialogMsg] = React.useState("");
    const [enableWebcam, setEnableWebcam] = React.useState(true);
    const [authVertification, setAuthVertification] = React.useState(false);
    const [state, setState] = React.useState({
        name: "",
        id: "",
        department: ""
    });


    const handleChangeDropdown = event => {
        var userid = event.target.value;
        var userinfo = getUserInfoFromID(userid);
        setState(userinfo);
    };

    const getUserInfoFromID = (userid) => {
        userid = String(userid);
        for (var index in staffListArray) {
            var item = staffListArray[index];
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


    const cameraRef = React.useRef(null);

    const timer = React.useRef();

    const getImage = () => {
        return cameraRef.current.getScreenshot();
    }

    const handleCaptureButton = () => {
        if (enableWebcam) {
            var imageSrc = getImage();
            setImage(imageSrc);
            setAuthVertification(true);
        }
        else {
            alert("カメラをオンにしてください。")
        }
    }

    const onAuthentication = () => {
        setEnableWebcam(false);
        setAuthVertification(false);
        if (!loading) {
            setSuccessAuth(false);
            setLoading(true);
        }
        authentication.register(image).then(res => {
            return res.json();
        }).then(function (data) {
            console.log(data);
            setSuccessAuth(true);
            setLoading(false);
            setEnableWebcam(false);
            if (data.errcode == 0) {
                // processing after the response from server
                var msg = data.msg;
                setDialogMsg(msg);
                console.log({msg:msg});
                handleOpenDialog();
            } else {
                alert("エラー:" + data.msg.toString());
                // re render parent
                props.rerenderParentCallback();
            }
        })
    }

    const handleOpenDialog = () => {
        setOpen(true);
    };

    const handleCloseDialog = () => {
        setOpen(false);
        setOpenSelectDialog(true);
    };

    const handleCloseSelectDialog = () => {
        setOpenSelectDialog(false);
        // TODO: send true label to server
        var body = state;
        if(state.id===''){
            alert("正解IDを選択してください。")
            setOpenSelectDialog(true);
            return;
        }
        body['img_file_name'] = dialogMsg['img_file_name'];
        body['wrong_id'] = dialogMsg['id'];
        body['wrong_name'] = dialogMsg['name'];
        console.log({body:body});
        authentication.postFeedback(body).then(function (res) {
            return res.json();
        }).then(function (data) {
            console.log(data);
            if(data.errcode!=-1){
                alert("正解ラベルを送信完了しました。")
            }
            else{
                alert("エラー: "+data.msg);
            }
        })
        // re render parent
        props.rerenderParentCallback();
    }

    const handleAuthenticated = () => {
        setOpen(false);
        console.log("handleAuthenticated");

        var body = state;
        body['img_file_name'] = dialogMsg['img_file_name'];
        body['wrong_id'] = dialogMsg['id'];
        body['wrong_name'] = dialogMsg['name'];
        body['id'] = dialogMsg['id'];
        body['name'] = dialogMsg['name'];
        console.log({body:body})
        authentication.postFeedback(body).then(function (res) {
            return res.json();
        }).then(function (data) {
            console.log(data);
            if(data.errcode!=-1){
                alert("正解ラベルを送信完了しました。")
            }
            else{
                alert("error: "+data.msg);
            }
        })
        // re render parent
        props.rerenderParentCallback();
    }

    const handleCancelButton = () => {
        setEnableWebcam(false);
        // re render parent
        props.rerenderParentCallback();
    }


    const renderAuthInfo = () => {
        return (
            <div>認証結果:{authenticationInfo}</div>
        )
    }

    const renderMessageDialog = () => {
        return (
            <Dialog
                open={open}
                aria-labelledby="alert-dialog-title"
                aria-describedby="alert-dialog-description"
            >
                <DialogTitle id="alert-dialog-title">{"認証結果"}</DialogTitle>
                <DialogContent>
                    <DialogContentText id="alert-dialog-description">
                        システムが認証したID: {dialogMsg.id != -1 ? dialogMsg.id : "認識できませんでした。"}
                    </DialogContentText>
                </DialogContent>
                <DialogActions>
                    <Button onClick={handleCloseDialog} color="primary">
                        間違い
          </Button>
                    <Button onClick={handleAuthenticated} color="primary" autoFocus>
                        正解
          </Button>
                </DialogActions>
            </Dialog>
        )
    }

    const closeAuthVer = () => {
        setAuthVertification(false);
    }

    const renderAuthVertificationDialog = () => {
        return (
            <Dialog
                open={authVertification}
                onClose={closeAuthVer}
                aria-labelledby="alert-dialog-title"
                aria-describedby="alert-dialog-description"
            >
                <DialogTitle id="alert-dialog-title">{"以下の画像で認証しますか?"}</DialogTitle>
                <DialogContent>
                    <DialogContentText id="alert-dialog-description">
                        {image ?
                            <img width="320" height="240" src={`${image}`} />
                            : <div>画像が存在しません。</div>}
                    </DialogContentText>
                </DialogContent>
                <DialogActions>
                    <Button onClick={closeAuthVer} color="primary">
                        いいえ、撮り直します。
          </Button>
                    <Button onClick={onAuthentication} color="primary" autoFocus>
                        はい、送信します。
          </Button>
                </DialogActions>
            </Dialog>
        )
    }


    const renderSelectIdDialog = () => {
        return (
            <div>
                <Dialog
                    open={openSelectDialog}
                    onClose={handleCloseSelectDialog}
                    aria-labelledby="alert-dialog-title"
                    aria-describedby="alert-dialog-description"
                >
                    <DialogTitle id="alert-dialog-title">{"正解ID選択"}</DialogTitle>
                    <DialogContent>
                        <DialogContentText id="alert-dialog-description">
                            <FormControl className={classes.formControl}>
                                <InputLabel id="controlled-open-select-label">ID</InputLabel>
                                <Select
                                    labelId="controlled-open-select-label"
                                    id="controlled-open-select"
                                    value={state.id}
                                    onChange={handleChangeDropdown}
                                >
                                    {staffListArray.map(staff => (
                                        <MenuItem key={staff.id} value={staff.id}>
                                            {staff.id}
                                        </MenuItem>
                                    ))}
                                </Select>
                            </FormControl>
                        </DialogContentText>
                    </DialogContent>
                    <DialogActions>
                        <Button onClick={handleCancelButton} color="primary">
                            キャンセル</Button>
                        <Button onClick={handleCloseSelectDialog} color="primary" autoFocus>
                            送信</Button>
                    </DialogActions>
                </Dialog>
            </div >
        )
    }


    return (
        <Paper className={classes.paper}>
            <div className={classes.title}>
                ユーザー認証
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
                        ref={cameraRef}
                        screenshotFormat="image/jpeg"
                        width={videoConstraints.width}
                        videoConstraints={videoConstraints}
                    /> : ""}
                </Grid>
                <div>
                    <Button onClick={handleCaptureButton}
                        color="primary"
                        variant="contained"
                        className={classes.button}
                        disabled={loading}
                    >開始</Button>
                    <Button onClick={handleCancelButton}
                        color="primary"
                        variant="contained"
                        className={classes.button}
                        disabled={loading}
                    >キャンセル</Button>
                </div>
                {loading && <CircularProgress size={72} className={classes.buttonProgress} />}

                <Divider />
                {showAuthenticationInfo ? renderAuthInfo() : ""}

            </Grid>
            {renderMessageDialog()}
            {renderAuthVertificationDialog()}
            {renderSelectIdDialog()}
        </Paper>

    );
}

export default withRouter(Authentication);


