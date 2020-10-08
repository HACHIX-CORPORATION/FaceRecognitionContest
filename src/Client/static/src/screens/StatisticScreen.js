import React, { Component, useEffect, useCallback, memo } from "react";
import { makeStyles, useTheme } from "@material-ui/core/styles";
import Paper from "@material-ui/core/Paper";
import Button from "@material-ui/core/Button";
import Card from '@material-ui/core/Card';
import CardActions from '@material-ui/core/CardActions';
import Grid from '@material-ui/core/Grid';
import { withRouter } from "react-router";
import CssBaseline from "@material-ui/core/CssBaseline";
import Container from "@material-ui/core/Container";
import { useTranslation } from "react-i18next";
// Local import
import authentication from "../data/authentication";

const useStyles = makeStyles(theme => ({
    root: {
        flexGrow: 1,
        overflow: "hidden",
        padding: theme.spacing(0, 3)
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
        align: "center",
        alignItems: "center"
    },
    title: {
        fontWeight: "bold",
        fontSize: "1.5rem",
        marginVertical: "1em",
        textAlign: "center"
    },
    media: {
        width: "90%",
        margin: theme.spacing(2),
        padding: theme.spacing(2)
    },
    space: {
        marginLeft: theme.spacing(3),
        marginRight: theme.spacing(3)
    }
}));

function StatisticScreen(props) {

    const { t, i18n } = useTranslation();
    const classes = useStyles();
    const [wrongList, setWrongList] = React.useState([]);
    const [correctProp, setCorrectProp] = React.useState(0);
    const [showSample, setShowSample] = React.useState(false);

    // load  for init
    useEffect(
        () => {
            authentication.getFeedback().then(res => {
                if (res.errcode != -1) {
                    setCorrectProp((res.msg.correct_ratio*100).toFixed(2));
                    var wrong_list = res.msg.wrong_list.sort(function (a, b){
                        return (a.file_name < b.file_name ? 1 : -1);
                    })
                    setWrongList(wrong_list);
                }
                else {
                    alert("error when get data from database");
                }
            })
        }, []);

    const onShowSamples = () => {
        setShowSample(!showSample);
    }

    const renderSamples = () => {
        var baseSrc = "/static/assets/FalseRecognition/"
        var n = Math.max(wrongList.length, 10);
        var sampleList = wrongList.slice(0, n);
        return (
            <Grid
                container
                direction="row"
                justify="flex-start"
                alignItems="flex-start">
                
                {sampleList.length>0?sampleList.map(item => {
                    var imgsrc = baseSrc + item.file_name;
                    return (
                        <Grid
                            key={item.file_name}
                            container
                            direction="row"
                            justify="flex-start"
                            alignItems="flex-start"
                            xs={4}
                        >
                            <img width="320px" height="240px" src={imgsrc}/>
                            <Grid container direction="row">
                                <h4 className={classes.space}>正解ID：{item.correct_id}</h4>
                                <h4 className={classes.space}>誤認識ID: {item.wrong_id}</h4>
                            </Grid>
                        </Grid>
                    )
                }):(<h3>誤認識画像が存在しません。</h3>)}
            </Grid>
        )
    }



    return (
        < div className={classes.root}>
            < div
                className={classes.title}>
                < h2> {t("認識性能指標")}</h2>
            </div>
            <CssBaseline />
            <Container fixed>
                <Paper className={classes.paper}>
                    <Card className={classes.root}>
                        <CardActions>
                            <Button color="primary" variant="contained" color="primary">
                                正解率: {correctProp}%
        </Button>
                        </CardActions>
                        <CardActions>
                            <Button color="primary" variant="contained" color="primary" onClick={onShowSamples}>
                                {!showSample ? "誤認識確認" : "誤認識を閉じる"}
                            </Button>
                        </CardActions>
                        {showSample ? renderSamples() : ""}
                    </Card>
                </Paper>
            </Container>
        </div>
    );
}

export default withRouter(StatisticScreen);


