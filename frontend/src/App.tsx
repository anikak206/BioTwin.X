import { createBrowserRouter, RouterProvider } from 'react-router-dom'
import Layout from './components/layout/Layout'
import Overview from './pages/Overview'
import Triage from './pages/Triage'
import PatientDetail from './pages/PatientDetail'
import Insights from './pages/Insights'
import Docs from './pages/Docs'

const router = createBrowserRouter([
  {
    path: '/',
    element: <Layout />,
    children: [
      { index: true, element: <Overview /> },
      { path: 'triage', element: <Triage /> },
      { path: 'patients', element: <Triage /> },
      { path: 'patients/:id', element: <PatientDetail /> },
      { path: 'insights', element: <Insights /> },
      { path: 'docs', element: <Docs /> },
    ],
  },
])

export default function App() {
  return <RouterProvider router={router} />
}
