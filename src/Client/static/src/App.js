import React, {Component} from "react";
import AppBar from "@material-ui/core/AppBar";
import Tabs from "@material-ui/core/Tabs";
import Tab from "@material-ui/core/Tab";

import {
    BrowserRouter as Router,
    Switch,
    Route,
    Link,
} from "react-router-dom";
import HomeScreen from "./screens/HomeScreen";
import StatisticScreen from "./screens/StatisticScreen";
import LanguageSelector from "./LanguageSelector";
import {withTranslation} from 'react-i18next';

function a11yProps(index) {
    return {
        id: `wrapped-tab-${index}`,
        "aria-controls": `wrapped-tabpanel-${index}`
    };
}

class MyApp extends Component {
    constructor(props) {
        super(props);
        this.state = {
            tab: "one"
        };
    }

    render() {
        const {t} = this.props;
        return (
            <div className="App">
                <Router>
                    <div>
                        <div>
                            <AppBar position="static">
                                <Tabs
                                    aria-label="simple tabs example"
                                    value={this.state.tab}
                                    onChange={(event, newValue) =>
                                        this.setState({tab: newValue})
                                    }
                                > <Tab
                                    label={t("Home")}
                                    value="one"
                                    to={"/"}
                                    component={Link}
                                    {...a11yProps("one")}
                                />
                                <Tab
                                    label={t("Performance")}
                                    value="two"
                                    to={"/statistic"}
                                    component={Link}
                                    {...a11yProps("two")}
                                    />
                                </Tabs>
                            </AppBar>
                        </div>

                        <Switch>
                            <Route exact path="/">
                                <HomeScreen/>
                            </Route>
                            <Route exact path="/statistic">
                                <StatisticScreen/>
                            </Route>
                        </Switch>
                    </div>
                </Router>
            </div>
        );
    }
}



export default withTranslation()(MyApp);