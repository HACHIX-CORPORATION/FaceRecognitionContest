import React, { Component, useEffect, useCallback, memo } from "react";
import { makeStyles, useTheme } from "@material-ui/core/styles";
import Paper from "@material-ui/core/Paper";
import Button from "@material-ui/core/Button";
import Divider from "@material-ui/core/Divider";
import { withRouter } from "react-router";
import CssBaseline from "@material-ui/core/CssBaseline";
import Container from "@material-ui/core/Container";
import { useTranslation } from "react-i18next";

// Local import
import Authentication from "../components/Authentication";
import Register from "../components/Register";

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
    }
}));

function HomeScreen(props) {

    const { t, i18n } = useTranslation();
    const classes = useStyles();
    const [registerMode, setRegister] = React.useState(false);
    const [authenticateMode, setAuth] = React.useState(false);

    // load  for init
    useEffect(
        () => {
        }, []);

    const videoConstraints = {
        width: 800,
        height: 600,
        facingMode: "user"
    };

    const register = () => {
        console.log("register mode");
        setAuth(false);
        setRegister(true);
    }

    const authenticate = () => {
        console.log("authenticate mode");
        setRegister(false);
        setAuth(true);
    }

    const rerenderParentCallback = () => {
        setAuth(false);
        setRegister(false);
    }



    return (
        < div className={classes.root}>
            < div
                className={classes.title}>
                < h2> {t("Talent5 Face Recognition Contest")}</h2>
            </div>
            <CssBaseline />
            <Container fixed>
                <Paper className={classes.paper}>
                    <div className={classes.title}>
                        <Button className={classes.button}
                            color="primary"
                            variant="contained"
                            onClick={register}
                        >Register</Button>
                        <Button className={classes.button}
                            color="primary"
                            variant="contained"
                            primary=""
                            onClick={authenticate}
                        >Recognition</Button>
                    </div>
                    <Divider />
                    {authenticateMode ? <Authentication settings={videoConstraints} rerenderParentCallback={rerenderParentCallback} /> : ""}
                    {registerMode ? <Register settings={videoConstraints} rerenderParentCallback={rerenderParentCallback} /> : ""}
                </Paper>
            </Container>
        </div>
    );
}

export default withRouter(HomeScreen);


