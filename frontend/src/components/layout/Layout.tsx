import Sidebar from './Sidebar'
import { Outlet } from 'react-router-dom'
import { Toaster } from 'sonner'

export default function Layout() {
  return (
    <div className="min-h-screen bg-background text-foreground">
      <Sidebar />
      <main className="ml-72 min-h-screen px-10 py-10">
        <Outlet />
      </main>
      <Toaster theme="dark" position="top-right" />
    </div>
  )
}
